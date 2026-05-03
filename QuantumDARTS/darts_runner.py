"""
QuantumDARTS runner for PSQASBench.

This runner keeps the search space faithful to the QuantumDARTS paper while
making the benchmark-facing outputs compatible with the rest of PSQASBench:

  - search slots use the paper-faithful macro candidates
      {Rz-Ry-Rz, I, CNOT(control -> target)}
  - benchmark evaluation/export lowers the final macro circuit to primitive
      {RZ, RY, CNOT} gates
  - direct-gate exports always carry optimised parameters in `angle`
"""

from __future__ import annotations

import configparser
import time
from pathlib import Path

import numpy as np
import torch

from RLQAS.base_runner import BaseRunner
from RLQAS.utils import set_seed

from .circuit import CNOT_NAME, IDENTITY_NAME, ROT_BLOCK_NAME
from .direct_opt import DirectGateBatchOptimizer, PreparedCircuit


_AXIS_TO_LABEL = {1: "RX", 2: "RY", 3: "RZ"}
_ROT_AXIS_SEQUENCE = (3, 2, 3)  # RZ -> RY -> RZ
_EVAL_TREND_FIELDS = [
    "eval_episode",
    "best_error_mha",
    "mean_error_mha",
    "mean_cnots",
    "D_struct",
    "D_func",
    "sr_at_chem",
    "cnot_at_chem",
]


class QuantumDARTSRunner(BaseRunner):
    """
    Run QuantumDARTS on one molecule / seed.

    The differentiable supernet search follows the paper candidate space.
    All benchmark-visible circuits are compiled to primitive gates and then
    optimised with the benchmark fixed-circuit optimizers.
    """

    def __init__(
        self,
        config_path: Path,
        mol_path: Path,
        result_dir: Path,
        seed: int,
        device: torch.device | None = None,
    ):
        self.config_path = Path(config_path)
        self.device = device or torch.device("cpu")

        cp = configparser.ConfigParser()
        cp.read(str(self.config_path))

        self.n_layers = cp.getint("env", "num_layers", fallback=15)
        self.accept_err = cp.getfloat("env", "accept_err", fallback=0.0016)
        self.analysis_save_threshold = cp.getfloat(
            "env",
            "analysis_save_threshold",
            fallback=self.accept_err,
        )
        self.connectivity = cp.get("env", "connectivity", fallback="all").strip().lower()
        self.max_gate_count = cp.getint("env", "max_gate_count", fallback=self.n_layers)

        self.n_epochs = cp.getint("general", "n_epochs", fallback=500)
        self.n_inner = cp.getint("general", "n_inner", fallback=5)
        self.eval_every = cp.getint("general", "eval_every", fallback=100)
        legacy_eval_k = cp.getint("general", "eval_k", fallback=8)
        self.trace_sample_count = cp.getint(
            "general",
            "trace_sample_count",
            fallback=legacy_eval_k,
        )
        self.eval_k = max(1, self.trace_sample_count)
        self.trace_attempt_factor = cp.getint("general", "trace_attempt_factor", fallback=5)
        self.compute_pcd = bool(cp.getint("general", "compute_pcd", fallback=1))

        self.lr_theta = cp.getfloat("arch", "lr_theta", fallback=0.01)
        self.lr_arch = cp.getfloat("arch", "lr_arch", fallback=0.005)
        self.tau_start = cp.getfloat("arch", "tau_start", fallback=1.0)
        self.tau_end = cp.getfloat("arch", "tau_end", fallback=0.05)
        self.K_prime = cp.getint("arch", "K_prime", fallback=3)
        self.gate_cost_weight = cp.getfloat("arch", "gate_cost_weight", fallback=0.0)

        legacy_finetune_steps = cp.getint("finetune", "finetune_steps", fallback=500)
        self.lr_finetune = cp.getfloat("finetune", "lr_finetune", fallback=0.005)
        self.discrete_eval_optim = cp.get(
            "non_local_opt",
            "optim_alg",
            fallback="COBYLA",
        ).strip().upper()
        self.discrete_eval_maxiter = cp.getint(
            "non_local_opt",
            "global_iters",
            fallback=legacy_finetune_steps,
        )
        self.discrete_eval_restarts = cp.getint(
            "non_local_opt",
            "n_restarts",
            fallback=1,
        )
        self.rotosolve_sweeps = cp.getint(
            "non_local_opt",
            "rotosolve_sweeps",
            fallback=1,
        )
        self.optim_method = cp.get(
            "non_local_opt",
            "method",
            fallback="scipy_each_step",
        ).strip()
        self.global_batched_cobyla = self._as_bool(
            cp.get("non_local_opt", "global_batched_cobyla", fallback="0"),
            default=False,
        )
        self.global_batched_rotosolve = self._as_bool(
            cp.get("non_local_opt", "global_batched_rotosolve", fallback="1"),
            default=True,
        )
        self.global_batched_spsa = self._as_bool(
            cp.get("non_local_opt", "global_batched_spsa", fallback="1"),
            default=True,
        )
        self.global_batched_psr = self._as_bool(
            cp.get("non_local_opt", "global_batched_psr", fallback="1"),
            default=True,
        )
        self.parallel_eval_batch_size = cp.getint(
            "non_local_opt",
            "parallel_eval_batch_size",
            fallback=0,
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
                "QuantumDARTS benchmark export currently supports only "
                f"method='scipy_each_step', got '{self.optim_method}'."
            )
        if self.discrete_eval_optim not in {"COBYLA", "ROTOSOLVE", "SPSA", "ADAMSPSA", "PSRADAM"}:
            raise ValueError(
                "QuantumDARTS benchmark export currently supports COBYLA, "
                "Rotosolve, SPSA, AdamSPSA, or PSRAdam "
                f"for fixed-circuit evaluation, got '{self.discrete_eval_optim}'."
            )

        conf: dict = {
            "env": {
                "accept_err": self.accept_err,
                "analysis_save_threshold": self.analysis_save_threshold,
                "connectivity": self.connectivity,
                "num_layers": self.n_layers,
            },
            "general": {"compute_pcd": int(self.compute_pcd)},
        }

        self._circuit: "QuantumDARTSCircuit | None" = None
        self._eval_history: list[dict] = []
        self._best_eval_metrics: dict | None = None
        self._best_eval_rollout: dict | None = None
        self._final_argmax_rollout: dict | None = None
        self._trace_rollouts: list[dict] = []

        super().__init__(conf, mol_path, result_dir, seed)

        self.n_qubits = int(round(np.log2(self.hamiltonian.shape[0])))
        self.H_tensor = torch.tensor(
            self.hamiltonian, dtype=torch.complex128, device=self.device
        )
        self._direct_optimizer = DirectGateBatchOptimizer(
            n_qubits=self.n_qubits,
            hamiltonian=self.hamiltonian,
            exact_energy=self.exact_energy,
            energy_shift=self.energy_shift,
            device=self.device,
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

    @staticmethod
    def _as_bool(v, default: bool = True) -> bool:
        if isinstance(v, bool):
            return v
        if v is None:
            return default
        if isinstance(v, (int, float)):
            return bool(int(v))
        text = str(v).strip().lower()
        if text in {"1", "true", "yes", "y", "on"}:
            return True
        if text in {"0", "false", "no", "n", "off"}:
            return False
        return default

    # ── Public runner interface ─────────────────────────────────────────────

    def run(self) -> dict:
        from .circuit import QuantumDARTSCircuit

        set_seed(self.seed, device=self.device)
        run_start = time.perf_counter()

        print(
            f"[QuantumDARTS] n_qubits={self.n_qubits}  slots={self.n_layers}  "
            f"seed={self.seed}  device={self.device}"
        )
        print(f"  candidate space = {{{ROT_BLOCK_NAME}, {IDENTITY_NAME}, {CNOT_NAME}}}")
        print(f"  connectivity    = {self.connectivity}")
        print(f"  max_gate_count  = {self.max_gate_count}")
        print(f"  eval optimizer  = {self.discrete_eval_optim}")
        if self.parallel_eval_batch_size > 0:
            print(f"  parallel batch  = {self.parallel_eval_batch_size}")
        print(f"  E_exact         = {self.exact_energy:.6f} Ha")
        print(f"  accept_err      = {self.accept_err * 1000:.1f} mHa")

        self._circuit = QuantumDARTSCircuit(
            self.n_qubits,
            self.n_layers,
            K_prime=self.K_prime,
            connectivity=self.connectivity,
        ).to(self.device)

        t0 = time.perf_counter()
        search_hist = self._search()
        t_search = time.perf_counter() - t0

        self._final_argmax_rollout = self.greedy_episode(None)
        self._trace_rollouts = self._collect_trace_rollouts()

        best_rollout = self._best_eval_rollout or self._final_argmax_rollout
        assert best_rollout is not None

        best_energy = float(best_rollout["energy"])
        error_ha = float(best_rollout["energy_error"])
        n_cnot = int(best_rollout["cnot_count"])
        depth = int(best_rollout["steps"])
        success = int(best_rollout["success"])
        nfev = int(best_rollout.get("nfev", -1))

        print(f"\n{'=' * 60}")
        print(f"  Best training-discovered energy = {best_energy:.6f} Ha")
        print(f"  Error                          = {error_ha * 1000:.4f} mHa")
        print(f"  Primitive gate count           = {depth}")
        print(f"  #CNOT                          = {n_cnot}")
        print(f"  Feasible under budget          = {bool(best_rollout.get('feasible', True))}")
        print(f"  Search time                    = {t_search:.1f}s")

        result = {
            "method": "QuantumDARTS",
            "seed": self.seed,
            "best_energy_ha": best_energy,
            "energy_error_ha": error_ha,
            "cnot_count": n_cnot,
            "circuit_depth": depth,
            "nfev": nfev,
            "success": success,
            "search_history": search_hist,
            "eval_best_error_mha": [m["best_error_mha"] for m in self._eval_history],
            "eval_sr_at_chem": [m["sr_at_chem"] for m in self._eval_history],
            "eval_d_struct": [m["D_struct"] for m in self._eval_history],
            "eval_d_func": [m["D_func"] for m in self._eval_history],
            "final_argmax_energy_ha": float(self._final_argmax_rollout["energy"]),
            "final_argmax_error_ha": float(self._final_argmax_rollout["energy_error"]),
            "final_argmax_depth": int(self._final_argmax_rollout["steps"]),
            "final_argmax_cnot_count": int(self._final_argmax_rollout["cnot_count"]),
            "exact_energy_ha": self.exact_energy,
            "accept_err_ha": self.accept_err,
        }

        npz_path = self.save_result(result)
        self._write_run_meta(t_search, time.perf_counter() - run_start)
        self._write_config_used_cfg()
        self._write_best_eval()
        self._write_discrete_eval_history()
        self._write_episode_traces()
        self._write_best_circuit()
        self._write_convergence_history(search_hist)
        print(f"  Saved -> {npz_path.parent}")
        return result

    def greedy_episode(self, env) -> dict:  # noqa: ARG002
        if self._circuit is None:
            raise RuntimeError("Call run() before greedy_episode().")
        gate_idx = self._circuit.discrete_gate_idx()
        return self._evaluate_gate_idx(gate_idx, seed=self.seed, source="argmax")

    def stochastic_episode(self, env, seed: int) -> dict:  # noqa: ARG002
        if self._circuit is None:
            raise RuntimeError("Call run() before stochastic_episode().")
        gate_idx = self._sample_gate_idx(seed)
        return self._evaluate_gate_idx(gate_idx, seed=seed, source="sampled")

    # ── Search loop ──────────────────────────────────────────────────────────

    def _search(self) -> list[float]:
        circuit = self._circuit
        assert circuit is not None

        opt_theta = torch.optim.Adam([circuit.theta], lr=self.lr_theta)
        opt_arch = torch.optim.Adam([circuit.P, circuit.Q], lr=self.lr_arch)
        sched_theta = torch.optim.lr_scheduler.CosineAnnealingLR(opt_theta, self.n_epochs)
        sched_arch = torch.optim.lr_scheduler.CosineAnnealingLR(opt_arch, self.n_epochs)

        history: list[float] = []
        for epoch in range(self.n_epochs):
            tau = self.tau_end + (self.tau_start - self.tau_end) * (
                1.0 - epoch / max(self.n_epochs, 1)
            )

            with torch.no_grad():
                h_fixed = circuit.sample_h(tau).detach()

            for _ in range(self.n_inner):
                loss_theta = circuit.energy_fixed_h(h_fixed, self.H_tensor)
                opt_theta.zero_grad()
                loss_theta.backward()
                opt_theta.step()

            loss_energy = circuit.energy(self.H_tensor, tau)
            loss_arch = loss_energy
            if self.gate_cost_weight > 0.0:
                loss_arch = loss_arch + self.gate_cost_weight * circuit.expected_gate_cost()

            opt_arch.zero_grad()
            loss_arch.backward()
            opt_arch.step()

            sched_theta.step()
            sched_arch.step()

            total_energy = float(loss_energy.item() + self.energy_shift)
            history.append(total_energy)

            if (epoch + 1) % 50 == 0 or epoch + 1 == self.n_epochs:
                err_mha = abs(total_energy - self.exact_energy) * 1000.0
                print(
                    f"  [epoch {epoch + 1:4d}/{self.n_epochs}]  "
                    f"E_soft={total_energy:.6f} Ha  err={err_mha:.2f} mHa  tau={tau:.3f}"
                )

            should_eval = self.eval_every > 0 and (
                (epoch + 1) % self.eval_every == 0 or epoch + 1 == self.n_epochs
            )
            if should_eval:
                self._run_eval_checkpoint(epoch + 1)

        return history

    # ── Architecture decoding / compilation ─────────────────────────────────

    def _argmax_gate_idx(self) -> torch.Tensor:
        circuit = self._circuit
        assert circuit is not None
        return circuit.discrete_gate_idx()

    def _sample_gate_idx(self, seed: int) -> torch.Tensor:
        circuit = self._circuit
        assert circuit is not None

        probs = circuit.arch_probs().detach().cpu().numpy()
        rng = np.random.default_rng(seed)
        gate_idx = np.zeros((self.n_layers, self.n_qubits), dtype=np.int64)

        for layer in range(self.n_layers):
            for target in range(self.n_qubits):
                n_choices = circuit.candidate_count(target)
                p = probs[layer, target, :n_choices]
                p = p / np.sum(p)
                gate_idx[layer, target] = int(rng.choice(n_choices, p=p))

        return torch.tensor(gate_idx, dtype=torch.long, device=self.device)

    @staticmethod
    def _gate_idx_key(gate_idx: torch.Tensor) -> tuple[int, ...]:
        return tuple(int(x) for x in gate_idx.detach().cpu().reshape(-1).tolist())

    def _gate_idx_to_macro_ops(self, gate_idx: torch.Tensor) -> list[dict]:
        circuit = self._circuit
        assert circuit is not None

        theta_np = circuit.theta.detach().cpu().numpy()
        gate_idx_np = gate_idx.detach().cpu().numpy()
        macro_ops: list[dict] = []

        for slot_layer in range(self.n_layers):
            for target in range(self.n_qubits):
                name, qubits = circuit.choice_spec(target, int(gate_idx_np[slot_layer, target]))
                if name == IDENTITY_NAME:
                    continue
                if name == ROT_BLOCK_NAME:
                    macro_ops.append(
                        {
                            "slot_layer": int(slot_layer),
                            "slot_qubit": int(target),
                            "macro_type": ROT_BLOCK_NAME,
                            "q": int(target),
                            "angles": [float(x) for x in theta_np[slot_layer, target]],
                        }
                    )
                else:
                    ctrl, targ = qubits
                    macro_ops.append(
                        {
                            "slot_layer": int(slot_layer),
                            "slot_qubit": int(target),
                            "macro_type": CNOT_NAME,
                            "ctrl": int(ctrl),
                            "targ": int(targ),
                        }
                    )

        return macro_ops

    def _compile_gate_idx(self, gate_idx: torch.Tensor) -> list[dict]:
        circuit = self._circuit
        assert circuit is not None

        theta_np = circuit.theta.detach().cpu().numpy()
        gate_idx_np = gate_idx.detach().cpu().numpy()
        compiled_ops: list[dict] = []
        primitive_layer = 0

        for slot_layer in range(self.n_layers):
            for target in range(self.n_qubits):
                name, qubits = circuit.choice_spec(target, int(gate_idx_np[slot_layer, target]))
                if name == IDENTITY_NAME:
                    continue

                if name == ROT_BLOCK_NAME:
                    angles = theta_np[slot_layer, target]
                    for macro_subindex, (axis, angle) in enumerate(zip(_ROT_AXIS_SEQUENCE, angles)):
                        compiled_ops.append(
                            {
                                "type": "rot",
                                "layer": int(primitive_layer),
                                "q": int(target),
                                "axis": int(axis),
                                "angle": float(angle),
                                "slot_layer": int(slot_layer),
                                "slot_qubit": int(target),
                                "macro_type": ROT_BLOCK_NAME,
                                "macro_subindex": int(macro_subindex),
                            }
                        )
                        primitive_layer += 1
                else:
                    ctrl, targ = qubits
                    compiled_ops.append(
                        {
                            "type": "cnot",
                            "layer": int(primitive_layer),
                            "ctrl": int(ctrl),
                            "targ": int(targ),
                            "slot_layer": int(slot_layer),
                            "slot_qubit": int(target),
                            "macro_type": CNOT_NAME,
                        }
                    )
                    primitive_layer += 1

        return compiled_ops

    @staticmethod
    def _apply_params_to_compiled_ops(compiled_ops: list[dict], params: np.ndarray) -> list[dict]:
        out: list[dict] = []
        param_idx = 0
        for op in compiled_ops:
            op_copy = dict(op)
            if op_copy["type"] == "rot":
                op_copy["angle"] = float(params[param_idx])
                param_idx += 1
            out.append(op_copy)
        return out

    def _compiled_ops_to_state_tensor(self, compiled_ops: list[dict]) -> tuple[torch.Tensor, int]:
        depth = len(compiled_ops)
        state_depth = max(1, depth)
        state = torch.zeros(state_depth, self.n_qubits + 6, self.n_qubits)

        for layer, op in enumerate(compiled_ops):
            if op["type"] == "rot":
                axis = int(op["axis"]) - 1
                q = int(op["q"])
                state[layer, self.n_qubits + axis, q] = 1.0
                state[layer, self.n_qubits + 3 + axis, q] = float(op["angle"])
            else:
                ctrl = int(op["ctrl"])
                targ = int(op["targ"])
                state[layer, targ, ctrl] = 1.0

        return state, depth

    # ── Fixed-circuit optimisation / evaluation ─────────────────────────────

    def _prepare_rollout_input(
        self,
        gate_idx: torch.Tensor,
        *,
        seed: int,
        source: str,
    ) -> dict:
        compiled_template = self._compile_gate_idx(gate_idx)
        macro_ops = self._gate_idx_to_macro_ops(gate_idx)
        state_tensor, _depth = self._compiled_ops_to_state_tensor(compiled_template)
        warm_start = np.asarray(
            [float(op["angle"]) for op in compiled_template if op["type"] == "rot"],
            dtype=np.float32,
        )
        return {
            "compiled_template": compiled_template,
            "macro_ops": macro_ops,
            "prepared": PreparedCircuit(
                state_tensor=state_tensor,
                warm_start=warm_start,
                seed=seed,
            ),
            "source": source,
        }

    def _build_rollout_from_prepared(self, info: dict, opt_result: dict) -> dict:
        compiled_ops = self._apply_params_to_compiled_ops(
            info["compiled_template"],
            np.asarray(opt_result["params"], dtype=np.float64),
        )
        state_tensor, depth = self._compiled_ops_to_state_tensor(compiled_ops)
        cnot_count = sum(1 for op in compiled_ops if op["type"] == "cnot")
        rotation_count = sum(1 for op in compiled_ops if op["type"] == "rot")
        feasible = self.max_gate_count <= 0 or depth <= self.max_gate_count
        success = int(float(opt_result["energy_error"]) < self.accept_err)
        return {
            "energy": float(opt_result["energy"]),
            "energy_error": float(opt_result["energy_error"]),
            "cnot_count": int(cnot_count),
            "rotation_count": int(rotation_count),
            "success": success,
            "steps": int(depth),
            "op_history": compiled_ops,
            "n_qubits": self.n_qubits,
            "final_state": state_tensor,
            "nfev": int(opt_result["nfev"]),
            "feasible": bool(feasible),
            "source": str(info["source"]),
            "macro_ops": [dict(op) for op in info["macro_ops"]],
        }

    def _evaluate_gate_indices(
        self,
        gate_indices: list[torch.Tensor],
        *,
        seeds: list[int],
        sources: list[str],
    ) -> list[dict]:
        prepared_infos = [
            self._prepare_rollout_input(gate_idx, seed=seed, source=source)
            for gate_idx, seed, source in zip(gate_indices, seeds, sources)
        ]
        opt_results = self._direct_optimizer.optimize_many(
            [info["prepared"] for info in prepared_infos]
        )
        return [
            self._build_rollout_from_prepared(info, opt_result)
            for info, opt_result in zip(prepared_infos, opt_results)
        ]

    def _evaluate_gate_idx(self, gate_idx: torch.Tensor, seed: int, source: str) -> dict:
        return self._evaluate_gate_indices(
            [gate_idx],
            seeds=[seed],
            sources=[source],
        )[0]

    @staticmethod
    def _trace_rollout_from_eval(rollout: dict, *, eval_episode: int, rollout_index: int) -> dict:
        return {
            "energy": float(rollout["energy"]),
            "energy_error": float(rollout["energy_error"]),
            "cnot_count": int(rollout["cnot_count"]),
            "rotation_count": int(rollout["rotation_count"]),
            "success": int(rollout["success"]),
            "steps": int(rollout["steps"]),
            "op_history": [dict(op) for op in rollout["op_history"]],
            "nfev": int(rollout.get("nfev", -1)),
            "feasible": bool(rollout.get("feasible", True)),
            "source": str(rollout.get("source", "sampled")),
            "macro_ops": [dict(op) for op in rollout.get("macro_ops", [])],
            "eval_episode": int(eval_episode),
            "rollout_index": int(rollout_index),
        }

    # ── Benchmark checkpoint evaluation ─────────────────────────────────────

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

    def _run_eval_checkpoint(self, eval_epoch: int) -> None:
        from metrics import aggregate_metrics, compute_pcd

        gate_indices = [self._argmax_gate_idx()]
        seeds = [self.seed]
        sources = ["argmax"]
        base_seed = self.seed * 100000 + eval_epoch * 100
        for idx in range(max(self.eval_k - 1, 0)):
            gate_indices.append(self._sample_gate_idx(base_seed + idx))
            seeds.append(base_seed + idx)
            sources.append("sampled")

        rollouts = self._evaluate_gate_indices(
            gate_indices,
            seeds=seeds,
            sources=sources,
        )
        argmax_rollout = rollouts[0]

        metrics = aggregate_metrics(rollouts, self.accept_err)
        best_rollout = self._select_best_rollout(rollouts)
        metrics["best_error_mha"] = float(best_rollout["energy_error"]) * 1000.0

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

        record = {
            **metrics,
            **pcd,
            "eval_episode": int(eval_epoch),
            "num_rollouts": len(rollouts),
            "num_feasible_rollouts": int(sum(1 for r in rollouts if r.get("feasible", True))),
            "argmax_energy_ha": float(argmax_rollout["energy"]),
            "argmax_error_ha": float(argmax_rollout["energy_error"]),
            "argmax_depth": int(argmax_rollout["steps"]),
            "best_rollout_energy": float(best_rollout["energy"]),
            "best_rollout_energy_error": float(best_rollout["energy_error"]),
            "best_rollout_depth": int(best_rollout["steps"]),
            "best_rollout_cnot_count": int(best_rollout["cnot_count"]),
            "best_rollout_rotation_count": int(best_rollout["rotation_count"]),
            "best_rollout_nfev": int(best_rollout.get("nfev", -1)),
            "best_rollout_feasible": int(bool(best_rollout.get("feasible", True))),
            "best_rollout_op_history": list(best_rollout["op_history"]),
        }
        self._eval_history.append(record)
        for rollout_index, rollout in enumerate(rollouts):
            self._trace_rollouts.append(
                self._trace_rollout_from_eval(
                    rollout,
                    eval_episode=eval_epoch,
                    rollout_index=rollout_index,
                )
            )

        if self._best_eval_metrics is None or self._rollout_is_better(
            best_rollout,
            self._best_eval_rollout,
        ):
            self._best_eval_metrics = dict(record)
            self._best_eval_rollout = {
                "energy": float(best_rollout["energy"]),
                "energy_error": float(best_rollout["energy_error"]),
                "cnot_count": int(best_rollout["cnot_count"]),
                "rotation_count": int(best_rollout["rotation_count"]),
                "success": int(best_rollout["success"]),
                "steps": int(best_rollout["steps"]),
                "op_history": [dict(op) for op in best_rollout["op_history"]],
                "n_qubits": int(best_rollout["n_qubits"]),
                "final_state": best_rollout["final_state"].clone(),
                "nfev": int(best_rollout.get("nfev", -1)),
                "feasible": bool(best_rollout.get("feasible", True)),
                "source": str(best_rollout.get("source", "sampled")),
                "macro_ops": [dict(op) for op in best_rollout.get("macro_ops", [])],
            }

        print(
            f"    [eval @ epoch {eval_epoch:4d}] "
            f"best={record['best_error_mha']:.3f} mHa  "
            f"argmax={record['argmax_error_ha'] * 1000.0:.3f} mHa  "
            f"SR={record['sr_at_chem']:.3f}  "
            f"D_struct={record['D_struct']:.4f}  D_func={record['D_func']:.4f}"
        )

    # ── Final trace pool ────────────────────────────────────────────────────

    def _collect_trace_rollouts(self) -> list[dict]:
        if self._trace_rollouts:
            return list(self._trace_rollouts)
        argmax_rollout = self._final_argmax_rollout or self.greedy_episode(None)
        return [
            self._trace_rollout_from_eval(
                argmax_rollout,
                eval_episode=self.n_epochs,
                rollout_index=0,
            )
        ]

    # ── Result writers ───────────────────────────────────────────────────────

    def _write_config_used_cfg(self) -> None:
        mol_file = Path(self.mol_path).name
        lines = [
            "[env]",
            f"num_qubits = {self.n_qubits}",
            f"accept_err = {self.accept_err}",
            f"analysis_save_threshold = {self.analysis_save_threshold}",
            f"connectivity = {self.connectivity}",
            f"max_gate_count = {self.max_gate_count}",
            "",
            "[problem]",
            f"mol_file = {mol_file}",
            "",
            "[general]",
            f"trace_sample_count = {self.trace_sample_count}",
            f"trace_attempt_factor = {self.trace_attempt_factor}",
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
        (self.result_dir / "config_used.cfg").write_text(
            "\n".join(lines) + "\n",
            encoding="utf-8",
        )

    def _write_run_meta(self, t_search: float, wall_clock_sec: float) -> None:
        best_eval_epoch = self._best_eval_metrics["eval_episode"] if self._best_eval_metrics else -1
        final_argmax = self._final_argmax_rollout or {}
        lines = [
            "method                     = QuantumDARTS",
            "variant                    = faithful_macro_search",
            f"mol_path                   = {self.mol_path}",
            f"seed                       = {self.seed}",
            f"exact_energy_ha            = {self.exact_energy:.8f}",
            f"accept_err_ha              = {self.accept_err:.6f}",
            f"analysis_save_threshold_ha = {self.analysis_save_threshold:.6f}",
            f"n_qubits                   = {self.n_qubits}",
            f"num_layers                 = {self.n_layers}",
            f"connectivity               = {self.connectivity}",
            f"max_gate_count             = {self.max_gate_count}",
            f"n_epochs                   = {self.n_epochs}",
            f"n_inner                    = {self.n_inner}",
            f"K_prime                    = {self.K_prime}",
            f"lr_theta                   = {self.lr_theta}",
            f"lr_arch                    = {self.lr_arch}",
            f"tau_start                  = {self.tau_start}",
            f"tau_end                    = {self.tau_end}",
            f"gate_cost_weight           = {self.gate_cost_weight}",
            f"eval_every                 = {self.eval_every}",
            f"eval_k                     = {self.eval_k}",
            f"trace_sample_count         = {self.trace_sample_count}",
            f"trace_attempt_factor       = {self.trace_attempt_factor}",
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
            f"legacy_lr_finetune         = {self.lr_finetune}",
            f"best_eval_epoch            = {best_eval_epoch}",
            f"search_time_s              = {t_search:.1f}",
            f"wall_clock_sec             = {wall_clock_sec:.1f}",
            f"final_argmax_error_mha     = {float(final_argmax.get('energy_error', float('nan'))) * 1000.0:.6f}",
            f"final_argmax_depth         = {int(final_argmax.get('steps', -1))}",
            f"final_argmax_feasible      = {int(bool(final_argmax.get('feasible', False)))}",
        ]
        for key, value in self.optim_options.items():
            lines.append(f"optim_option_{key}         = {value}")
        (self.result_dir / "run_meta.txt").write_text(
            "\n".join(lines) + "\n",
            encoding="utf-8",
        )

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
            f"D_struct = {self._best_eval_metrics.get('D_struct')}",
            f"D_func = {self._best_eval_metrics.get('D_func')}",
            f"num_rollouts = {self._best_eval_metrics.get('num_rollouts')}",
            f"num_feasible_rollouts = {self._best_eval_metrics.get('num_feasible_rollouts')}",
            "",
            "[best_eval_circuit]",
            f"energy_ha = {self._best_eval_metrics.get('best_rollout_energy')}",
            f"energy_error_ha = {self._best_eval_metrics.get('best_rollout_energy_error')}",
            f"depth = {self._best_eval_metrics.get('best_rollout_depth')}",
            f"cnot_count = {self._best_eval_metrics.get('best_rollout_cnot_count')}",
            f"rotation_count = {self._best_eval_metrics.get('best_rollout_rotation_count')}",
            f"nfev = {self._best_eval_metrics.get('best_rollout_nfev')}",
            f"feasible = {self._best_eval_metrics.get('best_rollout_feasible')}",
            f"op_history = {self._best_eval_metrics.get('best_rollout_op_history')!r}",
            "",
            "[eval_trend]",
            "\t".join(_EVAL_TREND_FIELDS),
        ]
        for row in self._eval_history:
            lines.append("\t".join(str(row.get(field)) for field in _EVAL_TREND_FIELDS))

        (self.result_dir / "best_eval.txt").write_text(
            "\n".join(lines) + "\n",
            encoding="utf-8",
        )

    def _write_discrete_eval_history(self) -> None:
        fields = [
            "eval_episode",
            "best_error_mha",
            "mean_error_mha",
            "sr_at_chem",
            "cnot_at_chem",
            "mean_cnots",
            "D_struct",
            "D_func",
            "best_rollout_depth",
            "best_rollout_cnot_count",
            "best_rollout_rotation_count",
            "best_rollout_feasible",
        ]
        lines = ["\t".join(fields)]
        for row in self._eval_history:
            lines.append("\t".join(str(row.get(field)) for field in fields))

        (self.result_dir / "discrete_eval_history.tsv").write_text(
            "\n".join(lines) + "\n",
            encoding="utf-8",
        )

    def _write_episode_traces(self) -> None:
        lines: list[str] = []
        for episode_idx, rollout in enumerate(self._trace_rollouts):
            snapshot = {
                "step": int(rollout.get("eval_episode", 0)),
                "eval_episode": int(rollout.get("eval_episode", 0)),
                "rollout_index": int(rollout.get("rollout_index", 0)),
                "source": rollout.get("source", "sampled"),
                "feasible": bool(rollout.get("feasible", True)),
                "gates_direct": [
                    {
                        key: (
                            int(value)
                            if isinstance(value, (int, np.integer))
                            else float(value)
                            if isinstance(value, (float, np.floating))
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

        (self.result_dir / "episode_traces.txt").write_text(
            "\n".join(lines),
            encoding="utf-8",
        )

    def _write_best_circuit(self) -> None:
        rollout = self._best_eval_rollout or self._final_argmax_rollout or self.greedy_episode(None)
        path = self.result_dir / "best_circuit.txt"
        macro_ops = rollout.get("macro_ops", [])
        compiled_ops = rollout.get("op_history", [])
        source_label = "training_best" if self._best_eval_rollout is not None else "final_argmax"

        lines = [
            f"source           = {source_label}",
            f"origin           = {rollout.get('source', 'argmax')}",
            f"energy_ha        = {float(rollout['energy']):.8f}",
            f"energy_error_ha  = {float(rollout['energy_error']):.8f}",
            f"energy_error_mha = {float(rollout['energy_error']) * 1000.0:.4f}",
            f"cnot_count       = {int(rollout['cnot_count'])}",
            f"rotation_count   = {int(rollout['rotation_count'])}",
            f"compiled_depth   = {int(rollout['steps'])}",
            f"max_gate_count   = {self.max_gate_count}",
            f"feasible         = {int(bool(rollout.get('feasible', True)))}",
            "",
            "--- macro circuit (paper-faithful search slots) ---",
        ]
        for op in macro_ops:
            if op["macro_type"] == ROT_BLOCK_NAME:
                a0, a1, a2 = op["angles"]
                lines.append(
                    f"  slot_layer={op['slot_layer']}  target={op['q']}  "
                    f"{ROT_BLOCK_NAME}({a0:.6f}, {a1:.6f}, {a2:.6f})"
                )
            else:
                lines.append(
                    f"  slot_layer={op['slot_layer']}  CNOT  ctrl={op['ctrl']}  targ={op['targ']}"
                )

        lines.append("")
        lines.append("--- compiled primitive circuit ---")
        for op in compiled_ops:
            if op["type"] == "rot":
                lines.append(
                    f"  layer={op['layer']}  {_AXIS_TO_LABEL[int(op['axis'])]}  "
                    f"q={op['q']}  angle={float(op['angle']):.6f}  "
                    f"(slot={op['slot_layer']}, macro={op.get('macro_subindex', -1)})"
                )
            else:
                lines.append(
                    f"  layer={op['layer']}  CNOT  ctrl={op['ctrl']}  targ={op['targ']}  "
                    f"(slot={op['slot_layer']})"
                )

        path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def _write_convergence_history(self, search_hist: list[float]) -> None:
        path = self.result_dir / "convergence_history.txt"
        with path.open("w", encoding="utf-8") as f:
            f.write("phase\tstep\tenergy_ha\tenergy_error_ha\n")
            for step, energy in enumerate(search_hist, start=1):
                f.write(f"search\t{step}\t{energy:.8f}\t{abs(energy - self.exact_energy):.8f}\n")
