"""
TF-QAS runner for PSQASBench.

Wraps the two-stage Training-Free QAS algorithm (He et al., AAAI 2024) as a
BaseRunner so it can be benchmarked alongside CRLQAS / HyRLQAS / QuantumDARTS.

Algorithm:
    Stage 1 (DAG path count):  sample S circuits, keep top-R by structural complexity.
    Stage 2 (Expressibility):  rank R circuits by KL divergence from Haar; keep top-K.
    Stage 3 (VQE):             COBYLA-optimise top-K circuits; return the best.

Unlike RL runners, TF-QAS has no training loop.
    greedy_episode  — evaluates the best VQE circuit from the last run().
    stochastic_episode(seed) — picks from top-K using seeded random (diversity for PCD).
"""

from __future__ import annotations

import configparser
import time
from pathlib import Path

import numpy as np
import torch

from RLQAS.base_runner import BaseRunner
from RLQAS.utils import set_seed


# rotation axis index: rx=0, ry=1, rz=2
_AXIS_MAP = {"rx": 0, "ry": 1, "rz": 2}
_PARAM_GATES = frozenset(["rx", "ry", "rz"])


class TFQASRunner(BaseRunner):
    """
    Run TF-QAS on one molecule / seed.

    Args:
        config_path : Path to .cfg in PSQASBench/configs/tfqas/
        mol_path    : Absolute path to the .npz Hamiltonian file
        result_dir  : Directory for result files
        seed        : Random seed
        device      : Ignored (TF-QAS runs on CPU only); kept for interface compat.
    """

    def __init__(
        self,
        config_path: Path,
        mol_path: Path,
        result_dir: Path,
        seed: int,
        device=None,   # noqa: ARG002 — interface compat, TF-QAS is CPU-only
    ):
        self.config_path = Path(config_path)

        # ── Parse config ──────────────────────────────────────────────────────
        cp = configparser.ConfigParser()
        cp.read(str(self.config_path))

        self.accept_err       = cp.getfloat("env",    "accept_err",       fallback=0.0016)
        self.n_layers         = cp.getint  ("search", "n_layers",         fallback=7)
        self.S                = cp.getint  ("search", "S",                fallback=10000)
        self.R                = cp.getint  ("search", "R",                fallback=1000)
        self.K                = cp.getint  ("search", "K",                fallback=20)
        self.n_expr_samples   = cp.getint  ("search", "n_expr_samples",   fallback=2000)
        self.n_bins           = cp.getint  ("search", "n_bins",           fallback=75)
        self.pipeline         = cp.get    ("search", "pipeline",         fallback="layerwise")
        self.n_workers        = cp.getint  ("search", "n_workers",        fallback=4)
        self.n_vqe_restarts   = cp.getint  ("vqe",   "n_vqe_restarts",   fallback=3)
        self.cobyla_maxiter   = cp.getint  ("vqe",   "cobyla_maxiter",   fallback=1000)
        _compute_pcd          = cp.getint  ("general","compute_pcd",      fallback=1)

        conf: dict = {
            "env":     {"accept_err": self.accept_err,
                        "num_layers": self.n_layers},
            "general": {"compute_pcd": str(_compute_pcd)},
        }

        # Populated by run()
        self._top_k: list | None = None   # TFQASResult list, sorted best VQE energy first

        super().__init__(conf, mol_path, result_dir, seed)

        self.n_qubits = int(round(np.log2(self.hamiltonian.shape[0])))

    # ── Abstract interface ─────────────────────────────────────────────────────

    def run(self) -> dict:
        """Full TF-QAS search (Stage 1 + 2 + VQE) and result serialisation."""
        from .tf_qas import run_tf_qas

        set_seed(self.seed)
        t0 = time.perf_counter()

        print(
            f"[TF-QAS] n_qubits={self.n_qubits}  n_layers={self.n_layers}  "
            f"seed={self.seed}"
        )
        print(f"  E_exact    = {self.exact_energy:.6f} Ha")
        print(f"  accept_err = {self.accept_err * 1000:.1f} mHa")
        print(f"  Budget: S={self.S}  R={self.R}  K={self.K}")

        mol = {
            "H":            self.hamiltonian,
            "energy_shift": self.energy_shift,
            "E_exact":      self.exact_energy,
        }

        results = run_tf_qas(
            n_qubits       = self.n_qubits,
            n_layers       = self.n_layers,
            S              = self.S,
            R              = self.R,
            K              = self.K,
            pipeline       = self.pipeline,
            n_expr_samples = self.n_expr_samples,
            n_bins         = self.n_bins,
            mol            = mol,
            n_vqe_restarts = self.n_vqe_restarts,
            cobyla_maxiter = self.cobyla_maxiter,
            seed           = self.seed,
            verbose        = True,
            n_workers      = self.n_workers,
        )

        t_total = time.perf_counter() - t0

        # Sort by VQE energy (best first)
        results_with_vqe = [r for r in results if r.vqe is not None]
        results_with_vqe.sort(key=lambda r: r.vqe.energy)
        self._top_k = results_with_vqe

        best = self._top_k[0]
        error_ha = best.vqe.error
        success  = int(error_ha < self.accept_err)

        print(f"\n{'='*55}")
        print(f"  Best energy  = {best.vqe.energy:.6f} Ha")
        print(f"  E_exact      = {self.exact_energy:.6f} Ha")
        print(f"  Error        = {error_ha * 1000:.4f} mHa  "
              f"{'✓ chem. accuracy' if success else '✗'}")
        print(f"  #CNOT        = {best.circuit.n_cnot}")
        print(f"  nfev (VQE)   = {best.vqe.n_fev}")
        print(f"  Time         = {t_total:.1f}s")

        result = {
            "method":           "TF-QAS",
            "seed":             self.seed,
            "best_energy_ha":   best.vqe.energy,
            "energy_error_ha":  error_ha,
            "cnot_count":       best.circuit.n_cnot,
            "circuit_depth":    self.n_layers,
            "nfev":             best.vqe.n_fev,
            "success":          success,
            "top_k_energies":   [r.vqe.energy for r in self._top_k],
            "top_k_errors_ha":  [r.vqe.error  for r in self._top_k],
            "top_k_cnot":       [r.circuit.n_cnot for r in self._top_k],
            "exact_energy_ha":  self.exact_energy,
            "accept_err_ha":    self.accept_err,
        }

        npz_path = self.save_result(result)
        self._write_run_meta(t_total)
        self._write_config_used_cfg()
        self._write_episode_traces(results_with_vqe, self.accept_err)
        self._write_best_circuit(best)
        self._write_search_summary()
        print(f"  Saved → {npz_path.parent}")
        return result

    def greedy_episode(self, env) -> dict:  # noqa: ARG002
        """Return rollout dict for the best VQE circuit (deterministic)."""
        if self._top_k is None:
            raise RuntimeError("Call run() before greedy_episode().")
        return self._result_to_rollout(self._top_k[0])

    def stochastic_episode(self, env, seed: int) -> dict:  # noqa: ARG002
        """
        Pick one circuit from top-K using seeded random (diversity for PCD).
        Different seeds yield different circuits from the top-K pool.
        `env` is accepted for BaseRunner interface compatibility; unused by TF-QAS.
        """
        if self._top_k is None:
            raise RuntimeError("Call run() before stochastic_episode().")
        rng = np.random.default_rng(seed)
        idx = int(rng.integers(0, len(self._top_k)))
        return self._result_to_rollout(self._top_k[idx])

    # ── Conversion helpers ─────────────────────────────────────────────────────

    def _result_to_rollout(self, tfqas_result) -> dict:
        """Convert a TFQASResult to the standard PSQASBench rollout-result dict."""
        circuit = tfqas_result.circuit
        params  = tfqas_result.vqe.params if tfqas_result.vqe is not None else None
        energy  = tfqas_result.vqe.energy if tfqas_result.vqe is not None else float("nan")
        error   = tfqas_result.vqe.error  if tfqas_result.vqe is not None else float("nan")

        state_tensor, op_history, depth = self._circuit_to_state_tensor(circuit, params)

        return {
            "energy":         energy,
            "energy_error":   error,
            "cnot_count":     circuit.n_cnot,
            "rotation_count": circuit.n_params,
            "success":        int(error < self.accept_err),
            "steps":          depth,
            "op_history":     op_history,
            "n_qubits":       self.n_qubits,
            "final_state":    state_tensor,
        }

    def _circuit_to_state_tensor(
        self,
        circuit,
        params: np.ndarray | None,
    ) -> tuple:
        """
        Convert TF-QAS Circuit to PSQASBench state tensor format.

        Uses a moments-based layer assignment (same logic as environment.py) so
        the resulting tensor is compatible with state_tensor_to_circuit / PCD.

        Returns
        -------
        state_tensor : torch.Tensor  shape (L, N+6, N)
        op_history   : list[dict]
        depth        : int   actual circuit depth L
        """
        N = self.n_qubits

        # Pass 1: compute actual depth via moments
        moments = [0] * N
        for gate_type, qubits in circuit.gates:
            if gate_type in _PARAM_GATES:
                q = qubits[0]
                moments[q] += 1
            else:  # CNOT
                ctrl, targ = qubits
                m = max(moments[ctrl], moments[targ])
                moments[ctrl] = m + 1
                moments[targ] = m + 1
        depth = max(moments) if any(m > 0 for m in moments) else 1

        state = torch.zeros(depth, N + 6, N)
        op_history = []
        moments = [0] * N
        p_idx = 0

        for gate_type, qubits in circuit.gates:
            if gate_type in _PARAM_GATES:
                q = qubits[0]
                layer = moments[q]
                axis  = _AXIS_MAP[gate_type]   # 0=RX, 1=RY, 2=RZ
                angle = float(params[p_idx]) if params is not None else 0.0
                state[layer, N + axis,     q] = 1.0
                state[layer, N + 3 + axis, q] = angle
                op_history.append({
                    "type": "rot", "layer": layer,
                    "q": q, "axis": axis + 1, "angle": angle,
                })
                moments[q] += 1
                p_idx += 1
            else:  # CNOT
                ctrl, targ = qubits
                layer = max(moments[ctrl], moments[targ])
                state[layer, targ, ctrl] = 1.0
                op_history.append({
                    "type": "cnot", "layer": layer,
                    "ctrl": ctrl, "targ": targ,
                })
                moments[ctrl] = layer + 1
                moments[targ] = layer + 1

        return state, op_history, depth

    # ── Result file writers ────────────────────────────────────────────────────

    def _write_config_used_cfg(self) -> None:
        """Write config_used.cfg required by critical_structure_tool.build_run_context()."""
        mol_file = Path(self.mol_path).name
        lines = [
            "[env]",
            f"num_qubits = {self.n_qubits}",
            f"accept_err = {self.accept_err}",
            "connectivity = all",
            "",
            "[problem]",
            f"mol_file = {mol_file}",
            "",
            "[non_local_opt]",
            "optim_alg = COBYLA",
            f"global_iters = {self.cobyla_maxiter}",
        ]
        (self.result_dir / "config_used.cfg").write_text(
            "\n".join(lines) + "\n", encoding="utf-8"
        )

    def _write_episode_traces(
        self,
        results_with_vqe: list,
        analysis_save_threshold_ha: float,
    ) -> None:
        """Write episode_traces.txt compatible with critical_structure_tool.

        Each circuit in *results_with_vqe* becomes one episode.  Episodes whose
        VQE error is at or below *analysis_save_threshold_ha* get an
        ``analysis_snapshot`` with ``gates_direct`` set so the tool can
        reconstruct the circuit without an action_dict.
        """
        lines: list[str] = []
        for ep_idx, result in enumerate(results_with_vqe):
            vqe = result.vqe
            if vqe is None:
                continue
            error_ha = float(vqe.error)
            _state, op_history, _depth = self._circuit_to_state_tensor(
                result.circuit, vqe.params
            )
            # op_history dicts are already compatible with gates_direct format;
            # the extra "layer" key is ignored by gate_dicts_to_gates().
            gates_direct = [
                {k: (int(v) if isinstance(v, (int, np.integer)) else float(v) if isinstance(v, (float, np.floating)) else v)
                 for k, v in op.items()}
                for op in op_history
            ]

            lines.append(f"[episode {ep_idx}]")
            lines.append(f"energy_errors_ha = [{error_ha!r}]")
            if error_ha <= analysis_save_threshold_ha:
                snapshot = {"step": 0, "gates_direct": gates_direct}
                lines.append(f"analysis_snapshots = [{snapshot!r}]")
            else:
                lines.append("analysis_snapshots = []")
            lines.append("")

        (self.result_dir / "episode_traces.txt").write_text(
            "\n".join(lines), encoding="utf-8"
        )

    def _write_run_meta(self, t_total: float):
        path = self.result_dir / "run_meta.txt"
        with open(path, "w") as f:
            f.write(f"method                     = TF-QAS\n")
            f.write(f"mol_path                   = {self.mol_path}\n")
            f.write(f"seed                       = {self.seed}\n")
            f.write(f"exact_energy_ha            = {self.exact_energy:.8f}\n")
            f.write(f"accept_err_ha              = {self.accept_err:.6f}\n")
            f.write(f"analysis_save_threshold_ha = {self.accept_err:.6f}\n")
            f.write(f"n_qubits                   = {self.n_qubits}\n")
            f.write(f"n_layers                   = {self.n_layers}\n")
            f.write(f"S                          = {self.S}\n")
            f.write(f"R                          = {self.R}\n")
            f.write(f"K                          = {self.K}\n")
            f.write(f"n_expr_samples             = {self.n_expr_samples}\n")
            f.write(f"pipeline                   = {self.pipeline}\n")
            f.write(f"n_vqe_restarts             = {self.n_vqe_restarts}\n")
            f.write(f"cobyla_maxiter             = {self.cobyla_maxiter}\n")
            f.write(f"total_time_s               = {t_total:.1f}\n")

    def _write_best_circuit(self, best_result):
        path = self.result_dir / "best_circuit.txt"
        vqe  = best_result.vqe
        with open(path, "w") as f:
            f.write(f"energy_ha        = {vqe.energy:.8f}\n")
            f.write(f"energy_error_ha  = {vqe.error:.8f}\n")
            f.write(f"energy_error_mha = {vqe.error * 1000:.4f}\n")
            f.write(f"cnot_count       = {best_result.circuit.n_cnot}\n")
            f.write(f"n_params         = {best_result.circuit.n_params}\n")
            f.write(f"success          = {int(vqe.error < self.accept_err)}\n")
            f.write(f"path_count       = {best_result.path_count}\n")
            f.write(f"expressibility   = {best_result.expressibility:.6f}\n")
            f.write(f"\n--- circuit gates ---\n")
            for gate_type, qubits in best_result.circuit.gates:
                if gate_type == "cnot":
                    f.write(f"  CNOT  ctrl={qubits[0]}  targ={qubits[1]}\n")
                else:
                    f.write(f"  {gate_type.upper():<4s}  q={qubits[0]}\n")
            f.write(f"\n--- optimised angles ---\n")
            if vqe.params is not None:
                for i, angle in enumerate(vqe.params):
                    f.write(f"  param[{i}] = {angle:.6f}\n")

    def _write_search_summary(self):
        """Write proxy scores and VQE results for all top-K circuits."""
        path = self.result_dir / "search_summary.txt"
        with open(path, "w") as f:
            f.write("rank\tpath_count\texpr\tn_cnot\tenergy_ha\terror_ha\tsuccess\n")
            for rank, r in enumerate(self._top_k, 1):
                vqe = r.vqe
                if vqe is not None:
                    f.write(
                        f"{rank}\t{r.path_count}\t{r.expressibility:.6f}\t"
                        f"{r.circuit.n_cnot}\t{vqe.energy:.8f}\t"
                        f"{vqe.error:.8f}\t{int(vqe.error < self.accept_err)}\n"
                    )
                else:
                    f.write(
                        f"{rank}\t{r.path_count}\t{r.expressibility:.6f}\t"
                        f"{r.circuit.n_cnot}\tN/A\tN/A\tN/A\n"
                    )
