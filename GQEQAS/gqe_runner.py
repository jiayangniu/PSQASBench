"""
GQERunner — offline-first, online-ready GQE runner for PSQASBench.

This implementation integrates a GPT-QE style method into the benchmark using
the standard result artefacts expected by downstream tooling:

  - result_seedXXXXX.npz
  - run_meta.txt
  - config_used.cfg
  - best_eval.txt
  - discrete_eval_history.tsv
  - episode_traces.txt
  - best_circuit.txt

The current implementation supports:
  - training_mode = offline | online
  - post-generation continuous re-optimization (training_reopt / benchmark_reopt)
  - UCC and primitive operator pools

The runner keeps a stable benchmark-visible output contract across both
training modes.
"""

from __future__ import annotations

import configparser
import time
from pathlib import Path

import numpy as np
import torch

from RLQAS.base_runner import BaseRunner
from metrics import aggregate_metrics, compute_pcd
from QuantumDARTS.direct_opt import DirectGateBatchOptimizer, PreparedCircuit

from .compiler import compile_ops_to_primitives
from .energy_eval import build_state_snapshot_circuit, get_subsequence_energies
from .model import GPTConfig, GPTQE
from .mol_adapter import OperatorPoolConfig, build_gqe_molecule


_AXIS_TO_LABEL = {1: "RX", 2: "RY", 3: "RZ"}
_EVAL_TREND_FIELDS = (
    "eval_episode",
    "best_error_mha",
    "mean_error_mha",
    "sr_at_chem",
    "cnot_at_chem",
    "mean_cnots",
    "pred_mae_mha",
    "best_rollout_depth",
    "best_rollout_cnot_count",
    "best_rollout_rotation_count",
    "best_rollout_token_len",
)


class GQERunner(BaseRunner):
    @staticmethod
    def _as_bool(value, *, default: bool = False) -> bool:
        if value is None:
            return default
        if isinstance(value, bool):
            return value
        text = str(value).strip().lower()
        if text in {"1", "true", "yes", "on"}:
            return True
        if text in {"0", "false", "no", "off"}:
            return False
        return default

    @staticmethod
    def _default_optimizer_config(mol_name: str) -> dict[str, object]:
        if any(tag in mol_name for tag in ("L1_BeH2_STO3G_6q", "L2_LiH_Equil_6q", "L4_H2_Stretch_4q")):
            return {
                "optim_alg": "COBYLA",
                "global_iters": 100,
                "n_restarts": 1,
                "rotosolve_sweeps": 1,
                "parallel_eval_batch_size": 0,
                "global_batched_cobyla": False,
                "global_batched_rotosolve": True,
                "global_batched_spsa": True,
                "global_batched_psr": True,
            }
        return {
            "optim_alg": "ROTOSOLVE",
            "global_iters": 100,
            "n_restarts": 1,
            "rotosolve_sweeps": 2,
            "parallel_eval_batch_size": 8,
            "global_batched_cobyla": False,
            "global_batched_rotosolve": True,
            "global_batched_spsa": True,
            "global_batched_psr": True,
        }

    def __init__(
        self,
        config_path: Path,
        mol_path: Path,
        result_dir: Path,
        seed: int,
        device=None,
    ):
        self.config_path = Path(config_path)
        cp = configparser.ConfigParser()
        cp.read(str(self.config_path))

        self.accept_err = cp.getfloat("env", "accept_err", fallback=0.0016)
        self.analysis_save_threshold = cp.getfloat(
            "env", "analysis_save_threshold", fallback=self.accept_err
        )
        self.n_qubits = cp.getint("env", "num_qubits")
        self.connectivity = cp.get("env", "connectivity", fallback="all").strip().lower()
        self.active_electrons = cp.getint("env", "active_electrons")
        self.active_orbitals = cp.getint("env", "active_orbitals", fallback=self.n_qubits // 2)

        self.seq_len = cp.getint("model", "seq_len", fallback=6)
        self.train_size = cp.getint("model", "train_size", fallback=2048)
        self.n_layer = cp.getint("model", "n_layer", fallback=6)
        self.n_head = cp.getint("model", "n_head", fallback=8)
        self.n_embd = cp.getint("model", "n_embd", fallback=256)
        self.dropout = cp.getfloat("model", "dropout", fallback=0.1)
        self.length_mode = cp.get("model", "length_mode", fallback="fixed").strip().lower()

        self.training_mode = cp.get("training", "training_mode", fallback="offline").strip().lower()
        self.epochs = cp.getint("training", "epochs", fallback=10000)
        self.lr = cp.getfloat("training", "lr", fallback=5e-5)
        self.batch_splits = cp.getint("training", "batch_splits", fallback=16)
        self.eval_every = cp.getint("training", "eval_every", fallback=max(1, self.epochs // 10))
        self.eval_n_sequences = cp.getint("training", "eval_n_sequences", fallback=100)
        self.eval_temperature = cp.getfloat("training", "eval_temperature", fallback=0.001)
        self.train_temperature = cp.getfloat("training", "train_temperature", fallback=1.0)
        self.online_sample_count = cp.getint("training", "online_sample_count", fallback=256)
        self.online_refresh_every = cp.getint("training", "online_refresh_every", fallback=20)
        self.online_temperature_final = cp.getfloat(
            "training",
            "online_temperature_final",
            fallback=self.eval_temperature,
        )
        self.online_temperature_schedule = cp.get(
            "training",
            "online_temperature_schedule",
            fallback="linear",
        ).strip().lower()
        self.replay_buffer_size = cp.getint("training", "replay_buffer_size", fallback=0)
        self.replay_mix_ratio = cp.getfloat("training", "replay_mix_ratio", fallback=0.0)
        self.warmup_epochs = cp.getint("training", "warmup_epochs", fallback=0)

        self.pool_kind = cp.get("operator_pool", "pool_kind", fallback="ucc").strip().lower()
        self.include_minus = bool(int(cp.get("operator_pool", "include_minus", fallback="1")))
        self.num_op_times = cp.getint("operator_pool", "num_op_times", fallback=8)
        self.op_time_min_exp = cp.getfloat("operator_pool", "op_time_min_exp", fallback=-2.0)
        self.op_time_max_exp = cp.getfloat("operator_pool", "op_time_max_exp", fallback=0.0)
        self.op_time_divisor = cp.getfloat("operator_pool", "op_time_divisor", fallback=160.0)

        self.compute_pcd = bool(int(cp.get("general", "compute_pcd", fallback="0")))
        reopt_default = 1 if self.pool_kind == "primitive" else 0
        self.training_reopt = bool(
            cp.getint("general", "training_reopt", fallback=reopt_default)
        )
        self.benchmark_reopt = bool(
            cp.getint("general", "benchmark_reopt", fallback=reopt_default)
        )

        non_local_defaults = self._default_optimizer_config(Path(mol_path).name)
        self.discrete_eval_optim = cp.get(
            "non_local_opt",
            "optim_alg",
            fallback=str(non_local_defaults["optim_alg"]),
        ).strip().upper()
        self.discrete_eval_maxiter = cp.getint(
            "non_local_opt",
            "global_iters",
            fallback=int(non_local_defaults["global_iters"]),
        )
        self.discrete_eval_restarts = cp.getint(
            "non_local_opt",
            "n_restarts",
            fallback=int(non_local_defaults["n_restarts"]),
        )
        self.rotosolve_sweeps = cp.getint(
            "non_local_opt",
            "rotosolve_sweeps",
            fallback=int(non_local_defaults["rotosolve_sweeps"]),
        )
        self.optim_method = cp.get(
            "non_local_opt",
            "method",
            fallback="scipy_each_step",
        ).strip()
        self.global_batched_cobyla = self._as_bool(
            cp.get(
                "non_local_opt",
                "global_batched_cobyla",
                fallback=str(int(non_local_defaults["global_batched_cobyla"])),
            ),
            default=bool(non_local_defaults["global_batched_cobyla"]),
        )
        self.global_batched_rotosolve = self._as_bool(
            cp.get(
                "non_local_opt",
                "global_batched_rotosolve",
                fallback=str(int(non_local_defaults["global_batched_rotosolve"])),
            ),
            default=bool(non_local_defaults["global_batched_rotosolve"]),
        )
        self.global_batched_spsa = self._as_bool(
            cp.get(
                "non_local_opt",
                "global_batched_spsa",
                fallback=str(int(non_local_defaults["global_batched_spsa"])),
            ),
            default=bool(non_local_defaults["global_batched_spsa"]),
        )
        self.global_batched_psr = self._as_bool(
            cp.get(
                "non_local_opt",
                "global_batched_psr",
                fallback=str(int(non_local_defaults["global_batched_psr"])),
            ),
            default=bool(non_local_defaults["global_batched_psr"]),
        )
        self.parallel_eval_batch_size = cp.getint(
            "non_local_opt",
            "parallel_eval_batch_size",
            fallback=int(non_local_defaults["parallel_eval_batch_size"]),
        )
        self.optim_options: dict[str, float] = {}
        if cp.has_option("non_local_opt", "a"):
            self.optim_options["a"] = cp.getfloat("non_local_opt", "a")
            for key in ("alpha", "c", "gamma", "beta_1", "beta_2", "lamda"):
                if cp.has_option("non_local_opt", key):
                    self.optim_options[key] = cp.getfloat("non_local_opt", key)
        elif cp.has_option("non_local_opt", "lr"):
            self.optim_options["lr"] = cp.getfloat("non_local_opt", "lr")
            for key in ("beta_1", "beta_2"):
                if cp.has_option("non_local_opt", key):
                    self.optim_options[key] = cp.getfloat("non_local_opt", key)

        if self.optim_method != "scipy_each_step":
            raise ValueError(
                "GQE currently only supports non_local_opt "
                f"method='scipy_each_step', got '{self.optim_method}'."
            )
        if self.discrete_eval_optim not in {"COBYLA", "ROTOSOLVE", "SPSA", "ADAMSPSA", "PSRADAM"}:
            raise ValueError(
                "GQE supports only the shared benchmark optimizers "
                f"{{COBYLA, ROTOSOLVE, SPSA, ADAMSPSA, PSRADAM}}, got '{self.discrete_eval_optim}'."
            )

        if device is not None and not isinstance(device, str):
            self.torch_device = str(device)
        else:
            self.torch_device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        variant = f"{self.training_mode}_gptqe__{self.pool_kind}_pool"
        self.variant = variant

        _conf = {
            "env": {
                "accept_err": self.accept_err,
                "analysis_save_threshold": self.analysis_save_threshold,
                "num_qubits": self.n_qubits,
                "connectivity": self.connectivity,
            },
            "general": {"compute_pcd": int(self.compute_pcd)},
        }
        super().__init__(_conf, mol_path, result_dir, seed)

        self._mol_data = None
        self._snapshot_circuit = None
        self._gpt = None
        self._eval_history: list[dict] = []
        self._all_rollouts: list[dict] = []
        self._best_eval_metrics: dict | None = None
        self._best_eval_rollout: dict | None = None
        self._best_train_rollout: dict | None = None
        self._best_global_rollout: dict | None = None
        self._best_model_epoch: int | None = None
        self._best_model_source: str | None = None
        self._direct_optimizer: DirectGateBatchOptimizer | None = None

    def run(self) -> dict:
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        run_start = time.perf_counter()

        print(
            f"[GQE] n_qubits={self.n_qubits}  seq_len={self.seq_len}  "
            f"seed={self.seed}  device={self.torch_device}",
            flush=True,
        )
        print(f"  training_mode   = {self.training_mode}", flush=True)
        print(f"  variant         = {self.variant}", flush=True)
        print(f"  pool_kind       = {self.pool_kind}", flush=True)
        print(f"  connectivity    = {self.connectivity}", flush=True)
        if self.training_mode == "offline":
            print(f"  train_size      = {self.train_size}", flush=True)
        else:
            print(f"  online_samples  = {self.online_sample_count}", flush=True)
            print(f"  refresh_every   = {self.online_refresh_every}", flush=True)
            print(
                f"  train_temp      = {self.train_temperature} -> {self.online_temperature_final} "
                f"({self.online_temperature_schedule})",
                flush=True,
            )
        print(f"  eval_n_sequences= {self.eval_n_sequences}", flush=True)
        print(f"  E_exact         = {self.exact_energy:.6f} Ha", flush=True)
        print(f"  accept_err      = {self.accept_err * 1000:.1f} mHa", flush=True)
        print(
            f"  eval optimizer  = {self.discrete_eval_optim}"
            + ("  (training+eval reopt)" if (self.training_reopt or self.benchmark_reopt) else "  (no_reopt)"),
            flush=True,
        )
        if self.parallel_eval_batch_size > 0:
            print(f"  parallel batch  = {self.parallel_eval_batch_size}", flush=True)

        pool_cfg = OperatorPoolConfig(
            pool_kind=self.pool_kind,
            include_minus=self.include_minus,
            num_op_times=self.num_op_times,
            op_time_min_exp=self.op_time_min_exp,
            op_time_max_exp=self.op_time_max_exp,
            op_time_divisor=self.op_time_divisor,
            connectivity=self.connectivity,
        )
        self._mol_data = build_gqe_molecule(
            self.mol_path,
            active_electrons=self.active_electrons,
            active_orbitals=self.active_orbitals,
            pool_cfg=pool_cfg,
        )
        self._snapshot_circuit = build_state_snapshot_circuit(
            self._mol_data["hf_state"],
            self._mol_data["n_qubits"],
        )
        if self.training_reopt or self.benchmark_reopt:
            self._direct_optimizer = DirectGateBatchOptimizer(
                n_qubits=self.n_qubits,
                hamiltonian=self.hamiltonian,
                exact_energy=self.exact_energy,
                energy_shift=self.energy_shift,
                device=torch.device(self.torch_device),
                optim_alg=self.discrete_eval_optim,
                global_iters=self.discrete_eval_maxiter,
                n_restarts=self.discrete_eval_restarts,
                rotosolve_sweeps=self.rotosolve_sweeps,
                options=self.optim_options,
                global_batched_cobyla=self.global_batched_cobyla,
                global_batched_rotosolve=self.global_batched_rotosolve,
                global_batched_spsa=self.global_batched_spsa,
                global_batched_psr=self.global_batched_psr,
                parallel_eval_batch_size=self.parallel_eval_batch_size,
            )

        model = GPTQE(
            GPTConfig(
                vocab_size=len(self._mol_data["op_pool"]) + 1,
                block_size=self.seq_len,
                n_layer=self.n_layer,
                n_head=self.n_head,
                n_embd=self.n_embd,
                dropout=self.dropout,
                bias=False,
            )
        ).to(self.torch_device)
        self._gpt = model
        optimizer = model.configure_optimizers(
            weight_decay=0.01,
            learning_rate=self.lr,
            betas=(0.9, 0.999),
        )

        t0 = time.perf_counter()
        if self.training_mode == "offline":
            losses = self._run_offline_training(model, optimizer)
        elif self.training_mode == "online":
            losses = self._run_online_training(model, optimizer)
        else:
            raise ValueError(f"Unsupported GQE training_mode '{self.training_mode}'.")
        total_time = time.perf_counter() - t0

        if not self._eval_history:
            print("[GQE] No periodic eval fired during training; running one final eval.", flush=True)
            self._evaluate_checkpoint(eval_episode=max(1, self.epochs), model=model)

        best_rollout = self._best_global_rollout or self._best_eval_rollout
        if best_rollout is None:
            raise RuntimeError("GQE run produced no evaluated rollouts.")

        result = {
            "method": "GQE",
            "variant": self.variant,
            "seed": self.seed,
            "best_energy_ha": float(best_rollout["energy"]),
            "energy_error_ha": float(best_rollout["energy_error"]),
            "cnot_count": int(best_rollout["cnot_count"]),
            "circuit_depth": int(best_rollout["steps"]),
            "token_len": int(best_rollout["token_len"]),
            "nfev": int(best_rollout.get("nfev", -1)),
            "success": int(best_rollout["success"]),
            "eval_best_error_mha": [row["best_error_mha"] for row in self._eval_history],
            "eval_sr_at_chem": [row["sr_at_chem"] for row in self._eval_history],
            "eval_mean_error_mha": [row["mean_error_mha"] for row in self._eval_history],
            "eval_mean_cnots": [row["mean_cnots"] for row in self._eval_history],
            "eval_pred_mae_mha": [row["pred_mae_mha"] for row in self._eval_history],
            "exact_energy_ha": self.exact_energy,
            "accept_err_ha": self.accept_err,
        }

        np.save(self.result_dir / "losses.npy", np.asarray(losses, dtype=np.float64))
        self.save_result(result)
        self._write_run_meta(total_time, time.perf_counter() - run_start)
        self._write_config_used_cfg()
        self._write_best_eval()
        self._write_discrete_eval_history()
        self._write_episode_traces()
        self._write_best_circuit()

        print(
            f"\n[GQE] Best benchmark energy = {float(best_rollout['energy']):.8f} Ha\n"
            f"[GQE] Error                 = {float(best_rollout['energy_error']) * 1000.0:.4f} mHa\n"
            f"[GQE] Compiled depth        = {int(best_rollout['steps'])}\n"
            f"[GQE] #CNOT                = {int(best_rollout['cnot_count'])}\n"
            f"[GQE] Saved -> {self.result_dir}",
            flush=True,
        )
        return result

    def greedy_episode(self, env) -> dict:  # noqa: ARG002
        if self._best_global_rollout is None and self._best_eval_rollout is None:
            raise RuntimeError("Call run() before greedy_episode().")
        return self._best_global_rollout or self._best_eval_rollout

    def stochastic_episode(self, env, seed: int) -> dict:  # noqa: ARG002
        if self._gpt is None:
            raise RuntimeError("Call run() before stochastic_episode().")
        return self._sample_rollouts(
            self._gpt,
            n_sequences=1,
            temperature=max(self.eval_temperature, 0.1),
            seed=seed,
        )[0]

    def _run_offline_training(self, model: GPTQE, optimizer: torch.optim.Optimizer) -> list[float]:
        tokens_t, energies_t = self._build_offline_dataset_tensors()
        losses: list[float] = []
        n_splits = max(1, min(int(self.batch_splits), int(tokens_t.shape[0])))

        print(
            f"[GQE] Offline training: epochs={self.epochs}  eval_every={self.eval_every}  "
            f"batch_splits={n_splits}",
            flush=True,
        )

        for epoch in range(self.epochs):
            model.train()
            epoch_loss = 0.0
            for tok_b, en_b in zip(torch.tensor_split(tokens_t, n_splits), torch.tensor_split(energies_t, n_splits)):
                optimizer.zero_grad()
                loss = model.calculate_loss(tok_b, en_b)
                loss.backward()
                optimizer.step()
                epoch_loss += float(loss.item())
            losses.append(epoch_loss)

            if (epoch + 1) % self.eval_every == 0:
                self._evaluate_checkpoint(eval_episode=epoch + 1, model=model)

        torch.save(model.state_dict(), self.result_dir / "final_model.pt")
        return losses

    def _run_online_training(self, model: GPTQE, optimizer: torch.optim.Optimizer) -> list[float]:
        losses: list[float] = []
        replay_tokens: np.ndarray | None = None
        replay_energies: np.ndarray | None = None
        current_tokens_t: torch.Tensor | None = None
        current_energies_t: torch.Tensor | None = None
        current_fresh_stats: tuple[float, float] | None = None
        current_source = "model"
        rng = np.random.default_rng(self.seed)

        print(
            f"[GQE] Online training: epochs={self.epochs}  eval_every={self.eval_every}  "
            f"online_sample_count={self.online_sample_count}  refresh_every={self.online_refresh_every}  "
            f"warmup_epochs={self.warmup_epochs}  replay={self.replay_buffer_size}/{self.replay_mix_ratio}",
            flush=True,
        )

        for epoch in range(self.epochs):
            refresh_every = max(1, int(self.online_refresh_every))
            refresh_batch = (epoch == 0) or (epoch % refresh_every == 0)
            if refresh_batch:
                target_batch = max(1, int(self.online_sample_count))
                target_replay = 0
                if self.replay_buffer_size > 0 and self.replay_mix_ratio > 0.0:
                    target_replay = int(round(target_batch * float(self.replay_mix_ratio)))
                replay_available = 0 if replay_tokens is None else int(replay_tokens.shape[0])
                replay_n = min(target_replay, replay_available)
                fresh_n = max(1, target_batch - replay_n)
                current_seq_len = self._draw_seq_len(rng)

                if epoch < self.warmup_epochs:
                    fresh_tokens_np, fresh_energies_np, fresh_rollouts = self._sample_random_training_batch(
                        n_sequences=fresh_n,
                        rng=rng,
                        current_seq_len=current_seq_len,
                    )
                    current_source = "random_warmup"
                    current_temp = float("nan")
                else:
                    current_temp = self._current_online_temperature(epoch)
                    fresh_tokens_np, fresh_energies_np, fresh_rollouts = self._sample_model_training_batch(
                        model,
                        n_sequences=fresh_n,
                        temperature=current_temp,
                        seed=self.seed * 100000 + epoch + 1,
                        current_seq_len=current_seq_len,
                    )
                    current_source = "model"

                if fresh_rollouts:
                    best_training_rollout = self._select_best_rollout(fresh_rollouts)
                    self._all_rollouts.append(best_training_rollout)
                    self._consider_best_rollout(
                        best_training_rollout,
                        model=model,
                        epoch=epoch + 1,
                    )

                if self.replay_buffer_size > 0:
                    replay_tokens, replay_energies = self._append_to_replay_buffer(
                        replay_tokens,
                        replay_energies,
                        fresh_tokens_np,
                        fresh_energies_np,
                    )

                train_tokens_np = fresh_tokens_np
                train_energies_np = fresh_energies_np
                if replay_n > 0 and replay_tokens is not None and replay_energies is not None:
                    replay_idx = rng.choice(replay_tokens.shape[0], size=replay_n, replace=False)
                    replay_tok_np = replay_tokens[replay_idx]
                    replay_en_np = replay_energies[replay_idx]
                    train_tokens_np = np.concatenate([train_tokens_np, replay_tok_np], axis=0)
                    train_energies_np = np.concatenate([train_energies_np, replay_en_np], axis=0)

                current_tokens_t = torch.from_numpy(train_tokens_np).to(self.torch_device)
                current_energies_t = torch.from_numpy(train_energies_np).float().to(self.torch_device)
                current_fresh_stats = (
                    float(fresh_energies_np[:, -1].min()),
                    float(fresh_energies_np[:, -1].mean()),
                )

                temp_label = "n/a" if not np.isfinite(current_temp) else f"{current_temp:.4f}"
                print(
                    f"  Refresh {epoch + 1:5d} | source {current_source} | temp {temp_label} | "
                    f"fresh {fresh_n} | replay {replay_n} | seq_len {current_seq_len} | "
                    f"best_err {abs(current_fresh_stats[0] - self.exact_energy) * 1000.0:.2f} mHa | "
                    f"mean_err {abs(current_fresh_stats[1] - self.exact_energy) * 1000.0:.2f} mHa",
                    flush=True,
                )

            if current_tokens_t is None or current_energies_t is None:
                raise RuntimeError("Online GQE failed to build a training batch.")

            model.train()
            epoch_loss = 0.0
            n_splits = max(1, min(int(self.batch_splits), int(current_tokens_t.shape[0])))
            for tok_b, en_b in zip(
                torch.tensor_split(current_tokens_t, n_splits),
                torch.tensor_split(current_energies_t, n_splits),
            ):
                optimizer.zero_grad()
                loss = model.calculate_loss(tok_b, en_b)
                loss.backward()
                optimizer.step()
                epoch_loss += float(loss.item())
            losses.append(epoch_loss)

            if (epoch + 1) % self.eval_every == 0:
                self._evaluate_checkpoint(eval_episode=epoch + 1, model=model)

        torch.save(model.state_dict(), self.result_dir / "final_model.pt")
        return losses

    def _build_offline_dataset_tensors(self) -> tuple[torch.Tensor, torch.Tensor]:
        rng = np.random.default_rng(self.seed)
        op_pool_size = len(self._mol_data["op_pool"])
        train_indices = rng.integers(op_pool_size, size=(self.train_size, self.seq_len))
        train_tokens = np.concatenate(
            [np.zeros((self.train_size, 1), dtype=int), train_indices + 1],
            axis=1,
        )
        train_op_seq = self._token_indices_to_ops(train_indices)
        print(f"[GQE] Sampling {self.train_size} offline training sequences…", flush=True)
        if self.training_reopt:
            _, train_energies = self._evaluate_sequences(
                token_rows=train_tokens,
                token_indices=train_indices,
                op_sequences=train_op_seq,
                predicted=np.full(self.train_size, np.nan, dtype=np.float64),
                seed_base=self.seed * 10000,
                source="offline_train",
                optimize=True,
            )
        else:
            train_energies = get_subsequence_energies(
                train_op_seq,
                snapshot_circuit=self._snapshot_circuit,
                hamiltonian_matrix=self._mol_data["hamiltonian_matrix"],
                energy_shift=self.energy_shift,
                log_interval=500,
            )
        print(
            f"[GQE] Training energy range: [{train_energies.min():.4f}, {train_energies.max():.4f}] Ha",
            flush=True,
        )
        tokens_t = torch.from_numpy(train_tokens).to(self.torch_device)
        energies_t = torch.from_numpy(train_energies).float().to(self.torch_device)
        return tokens_t, energies_t

    def _token_indices_to_ops(self, token_indices: np.ndarray) -> list[list]:
        op_pool = self._mol_data["op_pool"]
        return [[op_pool[int(i)] for i in row] for row in token_indices]

    def _sample_random_training_batch(
        self,
        *,
        n_sequences: int,
        rng: np.random.Generator,
        current_seq_len: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray, list[dict]]:
        op_pool_size = len(self._mol_data["op_pool"])
        n_tok = current_seq_len if current_seq_len is not None else self.seq_len
        token_indices = rng.integers(op_pool_size, size=(n_sequences, n_tok))
        token_rows = np.concatenate(
            [np.zeros((n_sequences, 1), dtype=int), token_indices + 1],
            axis=1,
        )
        op_sequences = self._token_indices_to_ops(token_indices)
        rollouts, energies = self._evaluate_sequences(
            token_rows=token_rows,
            token_indices=token_indices,
            op_sequences=op_sequences,
            predicted=np.full(n_sequences, np.nan, dtype=np.float64),
            seed_base=int(rng.integers(0, 2**31 - 1)),
            source="random_warmup",
            optimize=self.training_reopt,
        )
        token_rows_out, energies_out = self._pad_to_seq_len(
            token_rows.astype(np.int64, copy=False), energies
        )
        return token_rows_out, energies_out, rollouts

    def _sample_model_training_batch(
        self,
        model: GPTQE,
        *,
        n_sequences: int,
        temperature: float,
        seed: int,
        current_seq_len: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray, list[dict]]:
        torch.manual_seed(seed)
        np.random.seed(seed)

        token_rows, token_indices, predicted, op_sequences = self._sample_sequences_from_model(
            model,
            n_sequences=n_sequences,
            temperature=temperature,
            current_seq_len=current_seq_len,
        )
        rollouts, energies = self._evaluate_sequences(
            token_rows=token_rows,
            token_indices=token_indices,
            op_sequences=op_sequences,
            predicted=predicted,
            seed_base=seed,
            source="model_train",
            optimize=self.training_reopt,
        )
        token_rows, energies = self._pad_to_seq_len(token_rows, energies)
        return token_rows, energies, rollouts

    def _sample_sequences_from_model(
        self,
        model: GPTQE,
        *,
        n_sequences: int,
        temperature: float,
        current_seq_len: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[list]]:
        op_pool_size = len(self._mol_data["op_pool"])
        n_tok = current_seq_len if current_seq_len is not None else self.seq_len
        model.eval()
        with torch.no_grad():
            gen_tokens, pred_energies = model.generate(
                n_sequences=n_sequences,
                max_new_tokens=n_tok,
                temperature=temperature,
                device=self.torch_device,
            )
        token_rows = gen_tokens.cpu().numpy().astype(np.int64, copy=False)
        token_indices = np.clip(token_rows[:, 1:] - 1, 0, op_pool_size - 1)
        predicted = pred_energies.cpu().numpy().reshape(-1).astype(np.float64, copy=False)
        op_sequences = self._token_indices_to_ops(token_indices)
        return token_rows, token_indices, predicted, op_sequences

    def _append_to_replay_buffer(
        self,
        replay_tokens: np.ndarray | None,
        replay_energies: np.ndarray | None,
        fresh_tokens: np.ndarray,
        fresh_energies: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        if replay_tokens is None or replay_energies is None:
            replay_tokens = fresh_tokens.copy()
            replay_energies = fresh_energies.copy()
        else:
            replay_tokens = np.concatenate([replay_tokens, fresh_tokens], axis=0)
            replay_energies = np.concatenate([replay_energies, fresh_energies], axis=0)

        cap = max(0, int(self.replay_buffer_size))
        if cap > 0 and replay_tokens.shape[0] > cap:
            replay_tokens = replay_tokens[-cap:]
            replay_energies = replay_energies[-cap:]
        return replay_tokens, replay_energies

    def _current_online_temperature(self, epoch: int) -> float:
        start = float(self.train_temperature)
        end = float(self.online_temperature_final)
        schedule = self.online_temperature_schedule
        if schedule == "constant" or self.epochs <= 1:
            return start
        frac = float(epoch) / float(max(1, self.epochs - 1))
        if schedule == "linear":
            return start + (end - start) * frac
        if schedule == "cosine":
            return end + 0.5 * (start - end) * (1.0 + np.cos(np.pi * frac))
        raise ValueError(f"Unsupported GQE online_temperature_schedule '{schedule}'.")

    def _draw_seq_len(self, rng: np.random.Generator) -> int:
        """Sample circuit length: Uniform[1, seq_len] if length_mode='uniform', else seq_len."""
        if self.length_mode == "uniform":
            return int(rng.integers(1, self.seq_len + 1))
        return self.seq_len

    def _pad_to_seq_len(
        self,
        token_rows: np.ndarray,
        energies: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Pad token_rows and energies to seq_len for replay buffer shape consistency.

        Pads with BOS (token 0) so padded positions are never sampled during generation.
        Pads energies with the last real energy so cumulative logit targets are constant
        after the true circuit end.
        """
        current_len = token_rows.shape[1] - 1  # subtract BOS column
        if current_len >= self.seq_len:
            return token_rows, energies
        pad_cols = self.seq_len - current_len
        token_rows_padded = np.pad(token_rows, ((0, 0), (0, pad_cols)), constant_values=0)
        last_e = energies[:, -1:]
        energies_padded = np.concatenate([energies, np.tile(last_e, (1, pad_cols))], axis=1)
        return token_rows_padded, energies_padded

    def _consider_best_rollout(
        self,
        rollout: dict,
        *,
        model: GPTQE,
        epoch: int,
    ) -> None:
        if str(rollout.get("source", "")) != "eval":
            if self._best_train_rollout is None or self._rollout_is_better(rollout, self._best_train_rollout):
                self._best_train_rollout = rollout
        if self._best_global_rollout is None or self._rollout_is_better(rollout, self._best_global_rollout):
            self._best_global_rollout = rollout
            self._best_model_epoch = int(epoch)
            self._best_model_source = str(rollout.get("source", "unknown"))
            torch.save(model.state_dict(), self.result_dir / "best_model.pt")

    @staticmethod
    def _apply_params_to_compiled_ops(compiled_template: list[dict], params: np.ndarray) -> list[dict]:
        param_idx = 0
        out: list[dict] = []
        for layer, op in enumerate(compiled_template):
            op_copy = dict(op)
            op_copy["layer"] = int(layer)
            if op_copy["type"] == "rot":
                op_copy["angle"] = float(params[param_idx])
                param_idx += 1
            out.append(op_copy)
        return out

    def _prepare_sequence_info(
        self,
        *,
        ops: list,
        token_sequence: np.ndarray,
        predicted_energy: float,
        seed: int,
        source: str,
        eval_episode: int | None,
        eval_rollout_index: int,
    ) -> dict:
        compiled_template = compile_ops_to_primitives(ops)
        state_tensor, _depth = self._compiled_ops_to_state_tensor(compiled_template)
        warm_start = np.asarray(
            [float(op["angle"]) for op in compiled_template if op["type"] == "rot"],
            dtype=np.float32,
        )
        return {
            "compiled_template": compiled_template,
            "prepared": PreparedCircuit(
                state_tensor=state_tensor,
                warm_start=warm_start,
                seed=seed,
            ),
            "native_ops": list(ops),
            "token_sequence": [int(x) for x in token_sequence.tolist()],
            "predicted_energy": float(predicted_energy),
            "source": str(source),
            "eval_episode": int(eval_episode) if eval_episode is not None else -1,
            "eval_rollout_index": int(eval_rollout_index),
        }

    def _compiled_token_prefix_energies(self, compiled_ops: list[dict]) -> np.ndarray:
        if self._direct_optimizer is None:
            raise RuntimeError("Direct optimizer is not initialized for GQE prefix evaluation.")
        if not compiled_ops:
            return np.zeros((0,), dtype=np.float64)

        prefix_energies: list[float] = []
        current_prefix: list[dict] = []
        current_angles: list[float] = []
        for idx, op in enumerate(compiled_ops):
            current_prefix.append(op)
            if op["type"] == "rot":
                current_angles.append(float(op["angle"]))
            cur_token = int(op.get("source_token", idx))
            next_token = (
                int(compiled_ops[idx + 1].get("source_token", cur_token + 1))
                if idx + 1 < len(compiled_ops)
                else None
            )
            if next_token != cur_token:
                state_tensor, _ = self._compiled_ops_to_state_tensor(current_prefix)
                energy = self._direct_optimizer.evaluate_state(
                    state_tensor,
                    np.asarray(current_angles, dtype=np.float32),
                )
                prefix_energies.append(float(energy))
        return np.asarray(prefix_energies, dtype=np.float64)

    def _build_rollout_from_info(
        self,
        info: dict,
        *,
        compiled_ops: list[dict],
        full_energy: float,
        prefix_energies: np.ndarray,
        nfev: int,
    ) -> dict:
        state_tensor, depth = self._compiled_ops_to_state_tensor(compiled_ops)
        cnot_count = sum(1 for op in compiled_ops if op["type"] == "cnot")
        rotation_count = sum(1 for op in compiled_ops if op["type"] == "rot")
        error_ha = abs(float(full_energy) - self.exact_energy)
        return {
            "energy": float(full_energy),
            "predicted_energy": float(info["predicted_energy"]),
            "energy_error": float(error_ha),
            "cnot_count": int(cnot_count),
            "rotation_count": int(rotation_count),
            "success": int(error_ha < self.accept_err),
            "steps": int(depth),
            "token_len": int(len(info["token_sequence"])),
            "token_sequence": list(info["token_sequence"]),
            "op_history": compiled_ops,
            "native_ops": list(info["native_ops"]),
            "n_qubits": self.n_qubits,
            "final_state": state_tensor,
            "source": info["source"],
            "eval_episode": int(info["eval_episode"]),
            "eval_rollout_index": int(info["eval_rollout_index"]),
            "nfev": int(nfev),
            "prefix_energies": np.asarray(prefix_energies, dtype=np.float64),
        }

    def _evaluate_sequences(
        self,
        *,
        token_rows: np.ndarray,
        token_indices: np.ndarray,
        op_sequences: list[list],
        predicted: np.ndarray,
        seed_base: int,
        source: str,
        optimize: bool,
        eval_episode: int | None = None,
    ) -> tuple[list[dict], np.ndarray]:
        if optimize and self._direct_optimizer is None:
            raise RuntimeError("GQE optimize=True requested without an initialized direct optimizer.")

        if optimize:
            print(
                f"  [gqe {source}] optimizing {len(op_sequences)} circuits with {self.discrete_eval_optim}...",
                flush=True,
            )
            infos = [
                self._prepare_sequence_info(
                    ops=ops,
                    token_sequence=tok_idx,
                    predicted_energy=float(pred_e),
                    seed=int(seed_base + idx),
                    source=source,
                    eval_episode=eval_episode,
                    eval_rollout_index=idx,
                )
                for idx, (ops, tok_idx, pred_e) in enumerate(zip(op_sequences, token_indices, predicted))
            ]
            opt_results = self._direct_optimizer.optimize_many([info["prepared"] for info in infos])
            rollouts: list[dict] = []
            prefix_rows: list[np.ndarray] = []
            for info, opt_result in zip(infos, opt_results):
                compiled_ops = self._apply_params_to_compiled_ops(
                    info["compiled_template"],
                    np.asarray(opt_result["params"], dtype=np.float64),
                )
                prefix_energies = self._compiled_token_prefix_energies(compiled_ops)
                if prefix_energies.size:
                    prefix_energies[-1] = float(opt_result["energy"])
                rollout = self._build_rollout_from_info(
                    info,
                    compiled_ops=compiled_ops,
                    full_energy=float(opt_result["energy"]),
                    prefix_energies=prefix_energies,
                    nfev=int(opt_result["nfev"]),
                )
                rollouts.append(rollout)
                prefix_rows.append(prefix_energies)
            return rollouts, np.asarray(prefix_rows, dtype=np.float64)

        true_energy_matrix = get_subsequence_energies(
            op_sequences,
            snapshot_circuit=self._snapshot_circuit,
            hamiltonian_matrix=self._mol_data["hamiltonian_matrix"],
            energy_shift=self.energy_shift,
            log_interval=0,
        )
        rollouts = []
        for idx, (ops, tok_idx, pred_e, energy_row) in enumerate(
            zip(op_sequences, token_indices, predicted, true_energy_matrix)
        ):
            compiled = compile_ops_to_primitives(ops)
            rollout = self._build_rollout_from_info(
                {
                    "native_ops": list(ops),
                    "token_sequence": [int(x) for x in tok_idx.tolist()],
                    "predicted_energy": float(pred_e),
                    "source": str(source),
                    "eval_episode": int(eval_episode) if eval_episode is not None else -1,
                    "eval_rollout_index": int(idx),
                },
                compiled_ops=compiled,
                full_energy=float(energy_row[-1]),
                prefix_energies=np.asarray(energy_row, dtype=np.float64),
                nfev=0,
            )
            rollouts.append(rollout)
        return rollouts, np.asarray(true_energy_matrix, dtype=np.float64)

    def _evaluate_checkpoint(self, *, eval_episode: int, model: GPTQE) -> None:
        eval_rng = np.random.default_rng(self.seed + eval_episode)
        current_seq_len = self._draw_seq_len(eval_rng)
        token_rows, token_indices, predicted, op_sequences = self._sample_sequences_from_model(
            model,
            n_sequences=self.eval_n_sequences,
            temperature=self.eval_temperature,
            current_seq_len=current_seq_len,
        )
        rollouts, _ = self._evaluate_sequences(
            token_rows=token_rows,
            token_indices=token_indices,
            op_sequences=op_sequences,
            predicted=predicted,
            seed_base=self.seed * 1000 + eval_episode,
            source="eval",
            optimize=self.benchmark_reopt,
            eval_episode=eval_episode,
        )
        metrics = aggregate_metrics(rollouts, self.accept_err)
        best_rollout = self._select_best_rollout(rollouts)
        pred_mae_mha = float(np.mean([abs(r["predicted_energy"] - r["energy"]) for r in rollouts])) * 1000.0

        if self.compute_pcd:
            pcd = compute_pcd(
                rollouts,
                hamiltonian=self.hamiltonian,
                weights=self.weights,
                energy_shift=self.energy_shift,
                reoptimize=False,
            )
        else:
            pcd = {"D_struct": float("nan"), "D_func": float("nan"), "n_pairs": 0}

        row = {
            **metrics,
            **pcd,
            "eval_episode": int(eval_episode),
            "pred_mae_mha": pred_mae_mha,
            "num_rollouts": len(rollouts),
            "best_rollout_energy": float(best_rollout["energy"]),
            "best_rollout_energy_error": float(best_rollout["energy_error"]),
            "best_rollout_depth": int(best_rollout["steps"]),
            "best_rollout_cnot_count": int(best_rollout["cnot_count"]),
            "best_rollout_rotation_count": int(best_rollout["rotation_count"]),
            "best_rollout_token_len": int(best_rollout["token_len"]),
            "best_rollout_predicted_energy": float(best_rollout["predicted_energy"]),
            "best_rollout_op_history": [dict(op) for op in best_rollout["op_history"]],
        }
        self._eval_history.append(row)
        self._all_rollouts.extend(rollouts)

        if self._best_eval_rollout is None or self._rollout_is_better(best_rollout, self._best_eval_rollout):
            self._best_eval_rollout = best_rollout
            self._best_eval_metrics = dict(row)
        self._consider_best_rollout(best_rollout, model=model, epoch=eval_episode)

        print(
            f"  Eval {eval_episode:5d} | best_err {row['best_error_mha']:.2f} mHa | "
            f"mean_err {row['mean_error_mha']:.2f} mHa | SR {row['sr_at_chem']:.3f} | "
            f"CNOT {row['best_rollout_cnot_count']} | depth {row['best_rollout_depth']}",
            flush=True,
        )

    def _sample_rollouts(
        self,
        model: GPTQE,
        *,
        n_sequences: int,
        temperature: float,
        seed: int,
        eval_episode: int | None = None,
    ) -> list[dict]:
        torch.manual_seed(seed)
        np.random.seed(seed)
        token_rows, token_indices, predicted, op_sequences = self._sample_sequences_from_model(
            model,
            n_sequences=n_sequences,
            temperature=temperature,
        )
        rollouts, _ = self._evaluate_sequences(
            token_rows=token_rows,
            token_indices=token_indices,
            op_sequences=op_sequences,
            predicted=predicted,
            seed_base=seed,
            source="eval" if eval_episode is not None else "sample",
            optimize=self.benchmark_reopt,
            eval_episode=eval_episode,
        )
        return rollouts

    def _compiled_ops_to_state_tensor(self, compiled_ops: list[dict]) -> tuple[torch.Tensor, int]:
        depth = len(compiled_ops)
        state_depth = max(1, depth)
        state = torch.zeros(state_depth, self.n_qubits + 6, self.n_qubits)
        for layer, op in enumerate(compiled_ops):
            if op["type"] == "rot":
                q = int(op["q"])
                axis = int(op["axis"]) - 1
                state[layer, self.n_qubits + axis, q] = 1.0
                state[layer, self.n_qubits + 3 + axis, q] = float(op["angle"])
            else:
                ctrl = int(op["ctrl"])
                targ = int(op["targ"])
                state[layer, targ, ctrl] = 1.0
        return state, depth

    @staticmethod
    def _rollout_is_better(lhs: dict, rhs: dict) -> bool:
        lhs_key = (
            float(lhs["energy_error"]),
            int(lhs["steps"]),
            int(lhs["cnot_count"]),
        )
        rhs_key = (
            float(rhs["energy_error"]),
            int(rhs["steps"]),
            int(rhs["cnot_count"]),
        )
        return lhs_key < rhs_key

    def _select_best_rollout(self, rollouts: list[dict]) -> dict:
        best = rollouts[0]
        for rollout in rollouts[1:]:
            if self._rollout_is_better(rollout, best):
                best = rollout
        return best

    def _write_run_meta(self, total_time_s: float, wall_clock_sec: float) -> None:
        lines = [
            "method                     = GQE",
            f"variant                    = {self.variant}",
            f"mol_path                   = {self.mol_path}",
            f"seed                       = {self.seed}",
            f"exact_energy_ha            = {self.exact_energy:.8f}",
            f"accept_err_ha              = {self.accept_err:.6f}",
            f"analysis_save_threshold_ha = {self.analysis_save_threshold:.6f}",
            f"n_qubits                   = {self.n_qubits}",
            f"active_electrons           = {self.active_electrons}",
            f"active_orbitals            = {self.active_orbitals}",
            f"connectivity               = {self.connectivity}",
            f"training_mode              = {self.training_mode}",
            f"pool_kind                  = {self.pool_kind}",
            f"include_minus              = {int(self.include_minus)}",
            f"num_op_times               = {self.num_op_times}",
            f"seq_len                    = {self.seq_len}",
            f"train_size                 = {self.train_size}",
            f"n_layer                    = {self.n_layer}",
            f"n_head                     = {self.n_head}",
            f"n_embd                     = {self.n_embd}",
            f"dropout                    = {self.dropout}",
            f"length_mode                = {self.length_mode}",
            f"epochs                     = {self.epochs}",
            f"eval_every                 = {self.eval_every}",
            f"eval_n_sequences           = {self.eval_n_sequences}",
            f"eval_temperature           = {self.eval_temperature}",
            f"online_sample_count        = {self.online_sample_count}",
            f"online_refresh_every       = {self.online_refresh_every}",
            f"online_temperature_final   = {self.online_temperature_final}",
            f"online_temperature_schedule = {self.online_temperature_schedule}",
            f"replay_buffer_size         = {self.replay_buffer_size}",
            f"replay_mix_ratio          = {self.replay_mix_ratio}",
            f"warmup_epochs             = {self.warmup_epochs}",
            f"lr                         = {self.lr}",
            f"compute_pcd                = {int(self.compute_pcd)}",
            f"training_reopt             = {int(self.training_reopt)}",
            f"benchmark_reopt            = {int(self.benchmark_reopt)}",
            f"optim_method               = {self.optim_method}",
            f"discrete_eval_optim        = {self.discrete_eval_optim}",
            f"discrete_eval_maxiter      = {self.discrete_eval_maxiter}",
            f"discrete_eval_restarts     = {self.discrete_eval_restarts}",
            f"rotosolve_sweeps           = {self.rotosolve_sweeps}",
            f"parallel_eval_batch_size   = {self.parallel_eval_batch_size}",
            f"global_batched_cobyla      = {int(self.global_batched_cobyla)}",
            f"global_batched_rotosolve   = {int(self.global_batched_rotosolve)}",
            f"global_batched_spsa        = {int(self.global_batched_spsa)}",
            f"global_batched_psr         = {int(self.global_batched_psr)}",
            f"best_model_epoch           = {self._best_model_epoch if self._best_model_epoch is not None else -1}",
            f"best_model_source          = {self._best_model_source if self._best_model_source is not None else 'unknown'}",
            f"total_time_s               = {total_time_s:.1f}",
            f"wall_clock_sec             = {wall_clock_sec:.1f}",
        ]
        for key, value in self.optim_options.items():
            lines.append(f"{key:27s} = {value}")
        (self.result_dir / "run_meta.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    def _write_config_used_cfg(self) -> None:
        mol_file = Path(self.mol_path).name
        lines = [
            "[env]",
            f"num_qubits = {self.n_qubits}",
            f"accept_err = {self.accept_err}",
            f"analysis_save_threshold = {self.analysis_save_threshold}",
            f"active_electrons = {self.active_electrons}",
            f"active_orbitals = {self.active_orbitals}",
            f"connectivity = {self.connectivity}",
            "",
            "[problem]",
            f"mol_file = {mol_file}",
            "",
            "[operator_pool]",
            f"pool_kind = {self.pool_kind}",
            f"include_minus = {int(self.include_minus)}",
            f"num_op_times = {self.num_op_times}",
            f"op_time_min_exp = {self.op_time_min_exp}",
            f"op_time_max_exp = {self.op_time_max_exp}",
            f"op_time_divisor = {self.op_time_divisor}",
            "",
            "[model]",
            f"seq_len = {self.seq_len}",
            f"train_size = {self.train_size}",
            f"n_layer = {self.n_layer}",
            f"n_head = {self.n_head}",
            f"n_embd = {self.n_embd}",
            f"dropout = {self.dropout}",
            f"length_mode = {self.length_mode}",
            "",
            "[training]",
            f"training_mode = {self.training_mode}",
            f"epochs = {self.epochs}",
            f"lr = {self.lr}",
            f"batch_splits = {self.batch_splits}",
            f"eval_every = {self.eval_every}",
            f"eval_n_sequences = {self.eval_n_sequences}",
            f"eval_temperature = {self.eval_temperature}",
            f"train_temperature = {self.train_temperature}",
            f"online_sample_count = {self.online_sample_count}",
            f"online_refresh_every = {self.online_refresh_every}",
            f"online_temperature_final = {self.online_temperature_final}",
            f"online_temperature_schedule = {self.online_temperature_schedule}",
            f"replay_buffer_size = {self.replay_buffer_size}",
            f"replay_mix_ratio = {self.replay_mix_ratio}",
            f"warmup_epochs = {self.warmup_epochs}",
            "",
            "[general]",
            f"compute_pcd = {int(self.compute_pcd)}",
            f"training_reopt = {int(self.training_reopt)}",
            f"benchmark_reopt = {int(self.benchmark_reopt)}",
            "",
            "[non_local_opt]",
            f"method = {self.optim_method}",
            f"optim_alg = {self.discrete_eval_optim}",
        ]
        if self.discrete_eval_optim in {"COBYLA", "SPSA", "ADAMSPSA", "PSRADAM"}:
            lines.append(f"global_iters = {self.discrete_eval_maxiter}")
        if self.discrete_eval_optim == "ROTOSOLVE":
            lines.append(f"rotosolve_sweeps = {self.rotosolve_sweeps}")
        if self.discrete_eval_optim == "COBYLA" and self.discrete_eval_restarts:
            lines.append(f"n_restarts = {self.discrete_eval_restarts}")
        if self.parallel_eval_batch_size > 0:
            lines.append(f"parallel_eval_batch_size = {self.parallel_eval_batch_size}")
        lines.append(f"global_batched_cobyla = {int(self.global_batched_cobyla)}")
        lines.append(f"global_batched_rotosolve = {int(self.global_batched_rotosolve)}")
        lines.append(f"global_batched_spsa = {int(self.global_batched_spsa)}")
        lines.append(f"global_batched_psr = {int(self.global_batched_psr)}")
        for key, value in self.optim_options.items():
            lines.append(f"{key} = {value}")
        (self.result_dir / "config_used.cfg").write_text("\n".join(lines) + "\n", encoding="utf-8")

    def _write_best_eval(self) -> None:
        if self._best_eval_metrics is None:
            return
        lines = [
            "[best_eval_metrics]",
            f"eval_episode = {self._best_eval_metrics.get('eval_episode')}",
            f"sr_at_chem = {self._best_eval_metrics.get('sr_at_chem')}",
            f"cnot_at_chem = {self._best_eval_metrics.get('cnot_at_chem')}",
            f"best_error_mha = {self._best_eval_metrics.get('best_error_mha')}",
            f"mean_error_mha = {self._best_eval_metrics.get('mean_error_mha')}",
            f"mean_cnots = {self._best_eval_metrics.get('mean_cnots')}",
            f"pred_mae_mha = {self._best_eval_metrics.get('pred_mae_mha')}",
            f"D_struct = {self._best_eval_metrics.get('D_struct')}",
            f"D_func = {self._best_eval_metrics.get('D_func')}",
            f"num_rollouts = {self._best_eval_metrics.get('num_rollouts')}",
            "",
            "[best_eval_circuit]",
            f"energy_ha = {self._best_eval_metrics.get('best_rollout_energy')}",
            f"energy_error_ha = {self._best_eval_metrics.get('best_rollout_energy_error')}",
            f"depth = {self._best_eval_metrics.get('best_rollout_depth')}",
            f"cnot_count = {self._best_eval_metrics.get('best_rollout_cnot_count')}",
            f"rotation_count = {self._best_eval_metrics.get('best_rollout_rotation_count')}",
            f"token_len = {self._best_eval_metrics.get('best_rollout_token_len')}",
            f"predicted_energy_ha = {self._best_eval_metrics.get('best_rollout_predicted_energy')}",
            f"op_history = {self._best_eval_metrics.get('best_rollout_op_history')!r}",
            "",
            "[eval_trend]",
            "\t".join(_EVAL_TREND_FIELDS),
        ]
        for row in self._eval_history:
            lines.append("\t".join(str(row.get(field)) for field in _EVAL_TREND_FIELDS))
        (self.result_dir / "best_eval.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    def _write_discrete_eval_history(self) -> None:
        fields = list(_EVAL_TREND_FIELDS) + ["D_struct", "D_func"]
        lines = ["\t".join(fields)]
        for row in self._eval_history:
            lines.append("\t".join(str(row.get(field)) for field in fields))
        (self.result_dir / "discrete_eval_history.tsv").write_text(
            "\n".join(lines) + "\n",
            encoding="utf-8",
        )

    def _write_episode_traces(self) -> None:
        lines: list[str] = []
        for episode_idx, rollout in enumerate(self._all_rollouts):
            snapshot = {
                "step": 0,
                "eval_episode": int(rollout.get("eval_episode", -1)),
                "source": rollout.get("source", "eval"),
                "predicted_energy_ha": float(rollout.get("predicted_energy", float("nan"))),
                "token_sequence": list(rollout.get("token_sequence", [])),
                "gates_direct": [
                    {
                        key: (
                            int(value)
                            if isinstance(value, (int, np.integer))
                            else float(value)
                            if isinstance(value, (float, np.floating))
                            else list(value)
                            if isinstance(value, tuple)
                            else value
                        )
                        for key, value in op.items()
                    }
                    for op in rollout["op_history"]
                ],
            }
            lines.append(f"[episode {episode_idx}]")
            lines.append(f"energy_errors_ha = [{float(rollout['energy_error'])!r}]")
            lines.append(f"analysis_snapshots = [{snapshot!r}]")
            lines.append("")
        (self.result_dir / "episode_traces.txt").write_text("\n".join(lines), encoding="utf-8")

    def _write_best_circuit(self) -> None:
        if self._best_train_rollout is None and self._best_eval_rollout is None:
            return
        lines: list[str] = []

        def _append_rollout_block(title: str, rollout: dict) -> None:
            native_ops = rollout.get("native_ops", [])
            lines.extend([
                f"[{title}]",
                f"source           = {rollout.get('source', 'unknown')}",
                f"variant          = {self.variant}",
                f"energy_ha        = {float(rollout['energy']):.8f}",
                f"predicted_energy_ha = {float(rollout.get('predicted_energy', float('nan'))):.8f}",
                f"energy_error_ha  = {float(rollout['energy_error']):.8f}",
                f"energy_error_mha = {float(rollout['energy_error']) * 1000.0:.4f}",
                f"cnot_count       = {int(rollout['cnot_count'])}",
                f"rotation_count   = {int(rollout['rotation_count'])}",
                f"compiled_depth   = {int(rollout['steps'])}",
                f"token_len        = {int(rollout['token_len'])}",
                f"token_sequence   = {rollout.get('token_sequence', [])}",
                "",
                "--- native operator sequence ---",
            ])
            for idx, op in enumerate(native_ops):
                params = [float(p) for p in getattr(op, "parameters", ())]
                wires = [int(w) for w in getattr(op, "wires", [])]
                lines.append(
                    f"  token={idx:03d}  {str(getattr(op, 'name', type(op).__name__)):<24s} "
                    f"wires={wires}  params={params}"
                )
            lines.append("")
            lines.append("--- compiled primitive circuit ---")
            for op in rollout["op_history"]:
                layer = int(op.get("layer", -1))
                if op["type"] == "rot":
                    lines.append(
                        f"  layer={layer}  {_AXIS_TO_LABEL[int(op['axis'])]}  "
                        f"q={int(op['q'])}  angle={float(op['angle']):.6f}  "
                        f"(src={op.get('source_name')}, token={op.get('source_token')})"
                    )
                else:
                    lines.append(
                        f"  layer={layer}  CNOT  ctrl={int(op['ctrl'])}  targ={int(op['targ'])}  "
                        f"(src={op.get('source_name')}, token={op.get('source_token')})"
                    )
            lines.append("")

        if self._best_train_rollout is not None:
            _append_rollout_block("best_train_circuit", self._best_train_rollout)
        if self._best_eval_rollout is not None:
            _append_rollout_block("best_eval_circuit", self._best_eval_rollout)
        (self.result_dir / "best_circuit.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
