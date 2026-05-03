"""
TF-QAS runner for PSQASBench.

This wrapper keeps the training-free search logic close to the paper:

  - Stage 1: sample S native circuits and rank by DAG path count
  - Stage 2: rank the top-R circuits by expressibility
  - Stage 3 (benchmark wrapper): compile the top-K native circuits to primitive
    {RX, RY, RZ, CNOT} gates and optimise them with the shared benchmark
    fixed-circuit optimizer

Unlike RL runners, TF-QAS has no training loop.  Its benchmark output therefore
contains a single evaluation checkpoint over the final top-K candidate pool.
"""

from __future__ import annotations

import configparser
import time
from pathlib import Path

import numpy as np
import torch

from RLQAS.base_runner import BaseRunner
from RLQAS.utils import set_seed

from QuantumDARTS.direct_opt import DirectGateBatchOptimizer, PreparedCircuit


_AXIS_TO_LABEL = {1: "RX", 2: "RY", 3: "RZ"}
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


class TFQASRunner(BaseRunner):
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

        self.accept_err = cp.getfloat("env", "accept_err", fallback=0.0016)
        self.analysis_save_threshold = cp.getfloat(
            "env",
            "analysis_save_threshold",
            fallback=self.accept_err,
        )
        self.connectivity = cp.get("env", "connectivity", fallback="all").strip().lower()

        self.n_layers = cp.getint("search", "n_layers", fallback=7)
        self.S = cp.getint("search", "S", fallback=10000)
        self.R = cp.getint("search", "R", fallback=1000)
        self.K = cp.getint("search", "K", fallback=20)
        self.n_expr_samples = cp.getint("search", "n_expr_samples", fallback=2000)
        self.n_bins = cp.getint("search", "n_bins", fallback=75)
        self.pipeline = cp.get("search", "pipeline", fallback="gatewise").strip().lower()
        self.gate_set = cp.get("search", "gate_set", fallback="primitive").strip().lower()
        self.two_qubit_prob = cp.getfloat("search", "two_qubit_prob", fallback=0.5)
        self.n_workers = cp.getint("search", "n_workers", fallback=4)
        self.compute_pcd = bool(cp.getint("general", "compute_pcd", fallback=1))

        legacy_restarts = cp.getint("vqe", "n_vqe_restarts", fallback=3)
        legacy_maxiter = cp.getint("vqe", "cobyla_maxiter", fallback=1000)
        self.discrete_eval_optim = cp.get(
            "non_local_opt",
            "optim_alg",
            fallback="COBYLA",
        ).strip().upper()
        self.discrete_eval_maxiter = cp.getint(
            "non_local_opt",
            "global_iters",
            fallback=legacy_maxiter,
        )
        self.discrete_eval_restarts = cp.getint(
            "non_local_opt",
            "n_restarts",
            fallback=legacy_restarts,
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
                "TFQAS benchmark export currently supports only "
                f"method='scipy_each_step', got '{self.optim_method}'."
            )
        if self.discrete_eval_optim not in {"COBYLA", "ROTOSOLVE", "SPSA", "ADAMSPSA", "PSRADAM"}:
            raise ValueError(
                "TFQAS benchmark export currently supports COBYLA, "
                "Rotosolve, SPSA, AdamSPSA, or PSRAdam "
                f"for fixed-circuit evaluation, got '{self.discrete_eval_optim}'."
            )

        conf: dict = {
            "env": {
                "accept_err": self.accept_err,
                "analysis_save_threshold": self.analysis_save_threshold,
                "num_layers": self.n_layers,
            },
            "general": {"compute_pcd": int(self.compute_pcd)},
        }

        self._top_k: list = []
        self._eval_history: list[dict] = []
        self._best_eval_metrics: dict | None = None
        self._best_eval_rollout: dict | None = None
        self._rollouts: list[dict] = []

        super().__init__(conf, mol_path, result_dir, seed)

        self.n_qubits = int(round(np.log2(self.hamiltonian.shape[0])))
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

    def run(self) -> dict:
        from metrics import aggregate_metrics, compute_pcd
        from .tf_qas import run_tf_qas

        set_seed(self.seed, device=self.device)
        run_start = time.perf_counter()
        t0 = run_start

        print(
            f"[TF-QAS] n_qubits={self.n_qubits}  n_layers={self.n_layers}  "
            f"seed={self.seed}  device={self.device}"
        )
        print(f"  pipeline        = {self.pipeline}")
        _gate_label = (
            "{rx, ry, rz, cnot}" if self.gate_set == "primitive" else "{rx, ry, rz, xx, yy, zz}"
        )
        print(f"  candidate gates = {_gate_label}  (gate_set={self.gate_set})")
        if self.pipeline == "gatewise":
            print(f"  n_gates range   = [1, {self.n_layers}]  two_qubit_prob={self.two_qubit_prob}")
        print(f"  eval optimizer  = {self.discrete_eval_optim}")
        if self.parallel_eval_batch_size > 0:
            print(f"  parallel batch  = {self.parallel_eval_batch_size}")
        print(f"  E_exact         = {self.exact_energy:.6f} Ha")
        print(f"  accept_err      = {self.accept_err * 1000:.1f} mHa")
        print(f"  Budget: S={self.S}  R={self.R}  K={self.K}")

        self._top_k = run_tf_qas(
            n_qubits=self.n_qubits,
            n_layers=self.n_layers,
            S=self.S,
            R=self.R,
            K=self.K,
            pipeline=self.pipeline,
            n_expr_samples=self.n_expr_samples,
            n_bins=self.n_bins,
            mol=None,
            connectivity=self.connectivity,
            gate_mode=self.gate_set,
            two_qubit_prob=self.two_qubit_prob,
            seed=self.seed,
            verbose=True,
            n_workers=self.n_workers,
            bvqe=self._direct_optimizer._bvqe,
        )

        search_time = time.perf_counter() - t0
        eval_t0 = time.perf_counter()
        self._rollouts = self._evaluate_results(self._top_k)
        eval_time = time.perf_counter() - eval_t0

        metrics = aggregate_metrics(self._rollouts, self.accept_err)
        best_rollout = self._select_best_rollout(self._rollouts)
        metrics["best_error_mha"] = float(best_rollout["energy_error"]) * 1000.0

        if self.compute_pcd:
            pcd = compute_pcd(
                self._rollouts,
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
            "eval_episode": 1,
            "num_rollouts": len(self._rollouts),
            "num_feasible_rollouts": int(sum(1 for r in self._rollouts if r.get("feasible", True))),
            "best_rollout_energy": float(best_rollout["energy"]),
            "best_rollout_energy_error": float(best_rollout["energy_error"]),
            "best_rollout_depth": int(best_rollout["steps"]),
            "best_rollout_cnot_count": int(best_rollout["cnot_count"]),
            "best_rollout_rotation_count": int(best_rollout["rotation_count"]),
            "best_rollout_nfev": int(best_rollout.get("nfev", -1)),
            "best_rollout_feasible": int(bool(best_rollout.get("feasible", True))),
            "best_rollout_op_history": list(best_rollout["op_history"]),
        }
        self._eval_history = [record]
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
            "source": str(best_rollout.get("source", "topk")),
            "native_gates": [tuple(g) for g in best_rollout.get("native_gates", [])],
            "native_init_params": np.asarray(best_rollout.get("native_init_params", []), dtype=np.float64),
            "path_count": int(best_rollout.get("path_count", -1)),
            "expressibility": float(best_rollout.get("expressibility", float("nan"))),
        }

        best_energy = float(best_rollout["energy"])
        error_ha = float(best_rollout["energy_error"])
        n_cnot = int(best_rollout["cnot_count"])
        depth = int(best_rollout["steps"])
        success = int(best_rollout["success"])
        nfev = int(best_rollout.get("nfev", -1))

        print(f"\n{'=' * 60}")
        print(f"  Best benchmark energy        = {best_energy:.6f} Ha")
        print(f"  Error                        = {error_ha * 1000:.4f} mHa")
        print(f"  Primitive gate count         = {depth}")
        print(f"  #CNOT                        = {n_cnot}")
        print(f"  Feasible under budget        = {bool(best_rollout.get('feasible', True))}")
        print(f"  Search time                  = {search_time:.1f}s")
        print(f"  Primitive eval time          = {eval_time:.1f}s")

        result = {
            "method": "TF-QAS",
            "variant": f"{self.gate_set}_native_search__primitive_eval",
            "seed": self.seed,
            "best_energy_ha": best_energy,
            "energy_error_ha": error_ha,
            "cnot_count": n_cnot,
            "circuit_depth": depth,
            "nfev": nfev,
            "success": success,
            "eval_best_error_mha": [record["best_error_mha"]],
            "eval_sr_at_chem": [record["sr_at_chem"]],
            "eval_d_struct": [record["D_struct"]],
            "eval_d_func": [record["D_func"]],
            "top_k_path_count": [int(r.path_count) for r in self._top_k],
            "top_k_expressibility": [float(r.expressibility) for r in self._top_k],
            "exact_energy_ha": self.exact_energy,
            "accept_err_ha": self.accept_err,
        }

        npz_path = self.save_result(result)
        self._write_run_meta(search_time, eval_time, time.perf_counter() - run_start)
        self._write_config_used_cfg()
        self._write_best_eval()
        self._write_discrete_eval_history()
        self._write_episode_traces()
        self._write_best_circuit()
        self._write_search_summary()
        print(f"  Saved -> {npz_path.parent}")
        return result

    def greedy_episode(self, env) -> dict:  # noqa: ARG002
        if self._best_eval_rollout is None:
            raise RuntimeError("Call run() before greedy_episode().")
        return self._best_eval_rollout

    def stochastic_episode(self, env, seed: int) -> dict:  # noqa: ARG002
        if not self._rollouts:
            raise RuntimeError("Call run() before stochastic_episode().")
        rng = np.random.default_rng(seed)
        idx = int(rng.integers(0, len(self._rollouts)))
        return self._rollouts[idx]

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

    def _prepare_result_input(self, result, *, seed: int, source: str) -> dict:
        rng = np.random.default_rng(seed)
        native_init_params = result.circuit.initial_params(rng)
        compiled_template = result.circuit.compile_to_primitives(native_init_params)
        state_tensor, _depth = self._compiled_ops_to_state_tensor(compiled_template)
        warm_start = np.asarray(
            [float(op["angle"]) for op in compiled_template if op["type"] == "rot"],
            dtype=np.float32,
        )
        return {
            "compiled_template": compiled_template,
            "native_gates": list(result.circuit.gates),
            "native_init_params": native_init_params,
            "path_count": int(result.path_count),
            "expressibility": float(result.expressibility),
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
        return {
            "energy": float(opt_result["energy"]),
            "energy_error": float(opt_result["energy_error"]),
            "cnot_count": int(cnot_count),
            "rotation_count": int(rotation_count),
            "success": int(float(opt_result["energy_error"]) < self.accept_err),
            "steps": int(depth),
            "op_history": compiled_ops,
            "n_qubits": self.n_qubits,
            "final_state": state_tensor,
            "nfev": int(opt_result["nfev"]),
            "feasible": True,
            "source": str(info["source"]),
            "native_gates": list(info["native_gates"]),
            "native_init_params": np.asarray(info["native_init_params"], dtype=np.float64),
            "path_count": int(info["path_count"]),
            "expressibility": float(info["expressibility"]),
        }

    def _evaluate_results(self, results: list) -> list[dict]:
        K = len(results)
        print(f"\n[TF-QAS] Stage 3 — optimizing {K} circuits ...")
        prepared_infos = []
        for idx, result in enumerate(results):
            seed = self.seed * 100000 + idx
            prepared_infos.append(
                self._prepare_result_input(
                    result,
                    seed=seed,
                    source="topk",
                )
            )
        rollouts = []
        for idx, info in enumerate(prepared_infos):
            ct = info["compiled_template"]
            n_total = len(ct)
            n_cnot = sum(1 for op in ct if op["type"] == "cnot")
            n_rot = sum(1 for op in ct if op["type"] == "rot")
            n_params = info["prepared"].warm_start.size
            print(
                f"  {idx + 1}/{K}  depth={n_total}  cnot={n_cnot}  rot={n_rot}  params={n_params}",
                end="",
                flush=True,
            )
            t0 = time.perf_counter()
            opt_result = self._direct_optimizer.optimize_many([info["prepared"]])[0]
            elapsed = time.perf_counter() - t0
            err_mha = float(opt_result["energy_error"]) * 1000.0
            mark = "  (chem)" if float(opt_result["energy_error"]) < self.accept_err else ""
            print(f"  err={err_mha:.2f}mHa  [{elapsed:.1f}s]{mark}")
            rollouts.append(self._build_rollout_from_prepared(info, opt_result))
        return rollouts

    def _write_config_used_cfg(self) -> None:
        mol_file = Path(self.mol_path).name
        lines = [
            "[env]",
            f"num_qubits = {self.n_qubits}",
            f"accept_err = {self.accept_err}",
            f"analysis_save_threshold = {self.analysis_save_threshold}",
            f"connectivity = {self.connectivity}",
            "",
            "[problem]",
            f"mol_file = {mol_file}",
            "",
            "[search]",
            f"n_layers = {self.n_layers}",
            f"gate_set = {self.gate_set}",
            f"two_qubit_prob = {self.two_qubit_prob}",
            f"S = {self.S}",
            f"R = {self.R}",
            f"K = {self.K}",
            f"n_expr_samples = {self.n_expr_samples}",
            f"n_bins = {self.n_bins}",
            f"pipeline = {self.pipeline}",
            f"n_workers = {self.n_workers}",
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

    def _write_run_meta(self, search_time: float, eval_time: float, wall_clock_sec: float):
        best_eval_epoch = self._best_eval_metrics["eval_episode"] if self._best_eval_metrics else -1
        lines = [
            "method                     = TF-QAS",
            f"variant                    = {self.gate_set}_native_search__primitive_eval",
            f"mol_path                   = {self.mol_path}",
            f"seed                       = {self.seed}",
            f"exact_energy_ha            = {self.exact_energy:.8f}",
            f"accept_err_ha              = {self.accept_err:.6f}",
            f"analysis_save_threshold_ha = {self.analysis_save_threshold:.6f}",
            f"n_qubits                   = {self.n_qubits}",
            f"n_layers                   = {self.n_layers}",
            f"pipeline                   = {self.pipeline}",
            f"connectivity               = {self.connectivity}",
            f"S                          = {self.S}",
            f"R                          = {self.R}",
            f"K                          = {self.K}",
            f"n_expr_samples             = {self.n_expr_samples}",
            f"n_bins                     = {self.n_bins}",
            f"n_workers                  = {self.n_workers}",
            f"compute_pcd                = {int(self.compute_pcd)}",
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
            f"best_eval_epoch            = {best_eval_epoch}",
            f"search_time_s              = {search_time:.1f}",
            f"primitive_eval_time_s      = {eval_time:.1f}",
            f"wall_clock_sec             = {wall_clock_sec:.1f}",
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
        for episode_idx, rollout in enumerate(self._rollouts):
            snapshot = {
                # For direct-gate traces the critical-structure tool interprets
                # `step` as an index into `energy_errors_ha`.  TFQAS stores one
                # terminal error per candidate, so the snapshot index is 0.
                "step": 0,
                "eval_episode": 1,
                "source": rollout.get("source", "topk"),
                "path_count": int(rollout.get("path_count", -1)),
                "expressibility": float(rollout.get("expressibility", float("nan"))),
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
        rollout = self._best_eval_rollout or self.greedy_episode(None)
        path = self.result_dir / "best_circuit.txt"
        native_gates = rollout.get("native_gates", [])
        compiled_ops = rollout.get("op_history", [])
        native_init_params = np.asarray(rollout.get("native_init_params", []), dtype=np.float64)

        lines = [
            "source           = search_topk_best",
            f"origin           = {rollout.get('source', 'topk')}",
            f"energy_ha        = {float(rollout['energy']):.8f}",
            f"energy_error_ha  = {float(rollout['energy_error']):.8f}",
            f"energy_error_mha = {float(rollout['energy_error']) * 1000.0:.4f}",
            f"cnot_count       = {int(rollout['cnot_count'])}",
            f"rotation_count   = {int(rollout['rotation_count'])}",
            f"compiled_depth   = {int(rollout['steps'])}",
            f"path_count       = {int(rollout.get('path_count', -1))}",
            f"expressibility   = {float(rollout.get('expressibility', float('nan'))):.6f}",
            "",
            f"--- native {self.gate_set}-space circuit ---",
        ]

        for idx, (gate_type, qubits) in enumerate(native_gates):
            angle = float(native_init_params[idx]) if idx < native_init_params.shape[0] else float("nan")
            if len(qubits) == 1:
                lines.append(f"  {gate_type.upper():<4s} q={qubits[0]} init_angle={angle:.6f}")
            else:
                lines.append(
                    f"  {gate_type.upper():<4s} q0={qubits[0]} q1={qubits[1]} init_angle={angle:.6f}"
                )

        lines.append("")
        lines.append("--- compiled primitive circuit ---")
        for layer, op in enumerate(compiled_ops):
            layer_idx = int(op.get("layer", layer))
            if op["type"] == "rot":
                lines.append(
                    f"  layer={layer_idx}  {_AXIS_TO_LABEL[int(op['axis'])]}  "
                    f"q={op['q']}  angle={float(op['angle']):.6f}  "
                    f"(native={op.get('native_gate')}, idx={op.get('native_index', -1)})"
                )
            else:
                lines.append(
                    f"  layer={layer_idx}  CNOT  ctrl={op['ctrl']}  targ={op['targ']}  "
                    f"(native={op.get('native_gate')}, idx={op.get('native_index', -1)})"
                )

        path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def _write_search_summary(self):
        path = self.result_dir / "search_summary.txt"
        with path.open("w", encoding="utf-8") as f:
            f.write(
                "rank\tpath_count\texpr\tn_native_gates\tnative_two_qubit\t"
                "compiled_cnot\tenergy_ha\terror_ha\tsuccess\n"
            )
            for rank, (candidate, rollout) in enumerate(zip(self._top_k, self._rollouts), start=1):
                f.write(
                    f"{rank}\t{candidate.path_count}\t{candidate.expressibility:.6f}\t"
                    f"{len(candidate.circuit.gates)}\t{candidate.circuit.n_two_qubit}\t"
                    f"{rollout['cnot_count']}\t{rollout['energy']:.8f}\t"
                    f"{rollout['energy_error']:.8f}\t{int(rollout['energy_error'] < self.accept_err)}\n"
                )
