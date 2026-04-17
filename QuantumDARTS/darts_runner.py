"""
QuantumDARTS runner for PSQASBench.

Wraps the QuantumDARTS macro-search algorithm (Wu et al., ICML 2023) as a
BaseRunner so it can be benchmarked alongside CRLQAS / HyRLQAS.

Algorithm:
    Phase 1 (Architecture search): Alternate between
        - Inner θ updates with Gumbel-sampled h fixed
        - Outer α (P, Q) updates with a fresh differentiable sample
    Phase 2 (Fine-tune): Fix architecture (argmax of α), optimise θ with Adam.

Unlike RL runners, QuantumDARTS does not use CircuitEnv.  The Hamiltonian is
consumed directly as a dense matrix (already loaded by BaseRunner._load_molecule).

greedy_episode  — evaluates the argmax-discrete circuit (deterministic).
stochastic_episode — re-samples Gumbel with a given seed, may yield a
                     different discrete circuit (used for PCD diversity).
"""

from __future__ import annotations

import configparser
import time
from pathlib import Path

import numpy as np
import torch

from RLQAS.base_runner import BaseRunner
from RLQAS.utils import set_seed


# Gate name → rotation-axis index in PSQASBench state tensor (0=RX, 1=RY, 2=RZ)
_ROT_AXIS = {"Rx": 0, "Ry": 1, "Rz": 2}
_GATE_NAMES = ["Rx", "Ry", "Rz", "I", "CNOT"]


class QuantumDARTSRunner(BaseRunner):
    """
    Run QuantumDARTS on one molecule / seed.

    Args:
        config_path : Path to .cfg in PSQASBench/configs/qdarts/
        mol_path    : Absolute path to the .npz Hamiltonian file
        result_dir  : Directory for result files
        seed        : Random seed
        device      : torch.device (default: cpu)
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

        # ── Parse config ──────────────────────────────────────────────────────
        cp = configparser.ConfigParser()
        cp.read(str(self.config_path))

        self.n_layers       = cp.getint  ("env",      "num_layers",      fallback=15)
        self.accept_err     = cp.getfloat("env",      "accept_err",      fallback=0.0016)
        self.n_epochs       = cp.getint  ("general",  "n_epochs",        fallback=500)
        self.n_inner        = cp.getint  ("general",  "n_inner",         fallback=5)
        self.lr_theta       = cp.getfloat("arch",     "lr_theta",        fallback=0.01)
        self.lr_arch        = cp.getfloat("arch",     "lr_arch",         fallback=0.005)
        self.tau_start      = cp.getfloat("arch",     "tau_start",       fallback=1.0)
        self.tau_end        = cp.getfloat("arch",     "tau_end",         fallback=0.05)
        self.K_prime        = cp.getint  ("arch",     "K_prime",         fallback=3)
        self.finetune_steps = cp.getint  ("finetune", "finetune_steps",  fallback=500)
        self.lr_finetune    = cp.getfloat("finetune", "lr_finetune",     fallback=0.005)
        _compute_pcd        = cp.getint  ("general",  "compute_pcd",     fallback=1)

        # Build minimal config dict for BaseRunner compatibility
        conf: dict = {
            "env":     {"accept_err": self.accept_err,
                        "num_layers": self.n_layers},
            "general": {"compute_pcd": str(_compute_pcd)},
        }

        self._circuit: "QuantumDARTSCircuit | None" = None

        super().__init__(conf, mol_path, result_dir, seed)

        # Derived after _load_molecule()
        self.n_qubits = int(round(np.log2(self.hamiltonian.shape[0])))
        self.H_tensor = torch.tensor(
            self.hamiltonian, dtype=torch.complex128, device=self.device
        )

    # ── Abstract interface ────────────────────────────────────────────────────

    def run(self) -> dict:
        """Full two-phase QuantumDARTS training + result serialisation."""
        from .circuit import QuantumDARTSCircuit

        set_seed(self.seed)

        print(
            f"[QuantumDARTS] n_qubits={self.n_qubits}  n_layers={self.n_layers}  "
            f"seed={self.seed}  device={self.device}"
        )
        print(f"  E_exact = {self.exact_energy:.6f} Ha")
        print(f"  accept_err = {self.accept_err * 1000:.1f} mHa")

        # Build circuit
        self._circuit = QuantumDARTSCircuit(
            self.n_qubits, self.n_layers, K_prime=self.K_prime
        ).to(self.device)

        # Phase 1: Architecture search
        print(f"\n[Phase 1] Architecture search ({self.n_epochs} epochs, "
              f"n_inner={self.n_inner}) ...")
        t0 = time.perf_counter()
        search_hist = self._search()
        t_search = time.perf_counter() - t0
        print(f"  Search done  ({t_search:.1f}s)")

        # Phase 2: Fine-tune θ on discrete circuit
        print(f"\n[Phase 2] Fine-tuning θ ({self.finetune_steps} steps) ...")
        t1 = time.perf_counter()
        best_energy, finetune_hist = self._finetune()
        t_finetune = time.perf_counter() - t1

        # Extract discrete circuit and stats (use _gate_idx_to_list for 4-tuple format)
        with torch.no_grad():
            final_gate_idx = self._circuit.arch_weights().argmax(dim=-1)
        disc_gates = self._gate_idx_to_list(final_gate_idx)
        n_cnot     = sum(1 for _, g, _, _ in disc_gates if g == "CNOT")
        error_ha   = abs(best_energy - self.exact_energy)
        success    = int(error_ha < self.accept_err)
        t_total    = t_search + t_finetune

        print(f"\n{'='*55}")
        print(f"  Best energy  = {best_energy:.6f} Ha")
        print(f"  E_exact      = {self.exact_energy:.6f} Ha")
        print(f"  Error        = {error_ha * 1000:.4f} mHa  "
              f"{'✓ chem. accuracy' if success else '✗'}")
        print(f"  #CNOT        = {n_cnot}")
        print(f"  Time         = {t_total:.1f}s  "
              f"(search {t_search:.1f}s + finetune {t_finetune:.1f}s)")

        result = {
            "method":              "QuantumDARTS",
            "seed":                self.seed,
            "best_energy_ha":      best_energy,
            "energy_error_ha":     error_ha,
            "cnot_count":          n_cnot,
            "circuit_depth":       self.n_layers,
            "nfev":                -1,
            "success":             success,
            "search_history":      search_hist,
            "finetune_history":    finetune_hist,
            "cnot_history":        [],
            "exact_energy_ha":     self.exact_energy,
            "accept_err_ha":       self.accept_err,
        }

        npz_path = self.save_result(result)
        self._write_run_meta(t_total)
        self._write_best_circuit(best_energy, error_ha, n_cnot, disc_gates)
        self._write_convergence_history(search_hist, finetune_hist)
        print(f"  Saved → {npz_path.parent}")
        return result

    def greedy_episode(self, env) -> dict:  # noqa: ARG002
        """
        Evaluate the trained discrete circuit (argmax of α, deterministic).
        `env` is accepted for BaseRunner interface compatibility; unused by DARTS.
        """
        if self._circuit is None:
            raise RuntimeError("Call run() before greedy_episode().")
        with torch.no_grad():
            gate_idx = self._circuit.arch_weights().argmax(dim=-1)   # (L, N)
        return self._eval_with_gate_idx(gate_idx)

    def stochastic_episode(self, env, seed: int) -> dict:  # noqa: ARG002
        """
        Sample a Gumbel-stochastic circuit from trained α weights.
        Different seeds yield different Gumbel draws → different gate selections,
        providing the diversity needed for D_struct / D_func computation.
        `env` is accepted for BaseRunner interface compatibility; unused by DARTS.
        """
        if self._circuit is None:
            raise RuntimeError("Call run() before stochastic_episode().")

        torch.manual_seed(seed)
        np.random.seed(seed)

        with torch.no_grad():
            h = self._circuit.sample_h(tau=self.tau_end)
            gate_idx = h.argmax(dim=-1)    # (L, N) — seed-varied discrete circuit

        return self._eval_with_gate_idx(gate_idx)

    # ── Training phases ───────────────────────────────────────────────────────

    def _search(self) -> list:
        """Algorithm 1: alternate θ and α updates, return energy history."""
        circuit = self._circuit
        H       = self.H_tensor

        opt_theta = torch.optim.Adam([circuit.theta],        lr=self.lr_theta)
        opt_arch  = torch.optim.Adam([circuit.P, circuit.Q], lr=self.lr_arch)
        sched_t   = torch.optim.lr_scheduler.CosineAnnealingLR(opt_theta, self.n_epochs)
        sched_a   = torch.optim.lr_scheduler.CosineAnnealingLR(opt_arch,  self.n_epochs)

        history = []
        for epoch in range(self.n_epochs):
            tau = self.tau_end + (self.tau_start - self.tau_end) * (
                1.0 - epoch / self.n_epochs
            )

            # Inner loop: update θ with fixed Gumbel sample
            with torch.no_grad():
                h_fixed = circuit.sample_h(tau).detach()
            for _ in range(self.n_inner):
                loss = circuit.energy_fixed_h(h_fixed, H)
                opt_theta.zero_grad()
                loss.backward()
                opt_theta.step()

            # Outer loop: update architecture weights P, Q
            loss_arch = circuit.energy(H, tau)
            opt_arch.zero_grad()
            loss_arch.backward()
            opt_arch.step()

            sched_t.step()
            sched_a.step()

            total = loss_arch.item() + self.energy_shift
            history.append(total)

            if (epoch + 1) % 50 == 0:
                err_mha = abs(total - self.exact_energy) * 1000
                print(f"  [epoch {epoch+1:4d}/{self.n_epochs}]  "
                      f"E={total:.6f} Ha  err={err_mha:.2f} mHa  τ={tau:.3f}")

        return history

    def _finetune(self) -> tuple:
        """Fix architecture (argmax of α) and fine-tune θ with Adam.

        Returns
        -------
        best_energy : float
            Lowest energy found during fine-tuning (Ha).
        history : list[float]
            Energy at every fine-tune step (Ha), for convergence plotting.
        """
        circuit = self._circuit
        H       = self.H_tensor

        with torch.no_grad():
            alpha    = circuit.arch_weights()
            gate_idx = alpha.argmax(dim=-1)
            h_hard   = torch.zeros_like(alpha)
            for j in range(circuit.n_layers):
                for i in range(circuit.n_qubits):
                    h_hard[j, i, gate_idx[j, i]] = 1.0
        h_hard = h_hard.detach()

        opt     = torch.optim.Adam([circuit.theta], lr=self.lr_finetune)
        sched   = torch.optim.lr_scheduler.CosineAnnealingLR(opt, self.finetune_steps)
        best    = float("inf")
        history = []

        for step in range(self.finetune_steps):
            loss = circuit.energy_fixed_h(h_hard, H)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            total = loss.item() + self.energy_shift
            history.append(total)
            if total < best:
                best = total
            if (step + 1) % 100 == 0:
                err_mha = abs(total - self.exact_energy) * 1000
                print(f"  [finetune {step+1:4d}/{self.finetune_steps}]  "
                      f"E={total:.6f} Ha  err={err_mha:.2f} mHa")

        return best, history

    # ── Circuit helpers ───────────────────────────────────────────────────────

    def _gate_idx_to_list(self, gate_idx: torch.Tensor) -> list:
        """
        Convert (L, N) gate-index tensor to QuantumDARTS gate list:
            [(layer, gate_name, qubits_tuple, angle_or_None), ...]
        Identity gates are dropped.
        """
        theta_np = self._circuit.theta.detach().cpu().numpy()
        N        = self.n_qubits
        gates    = []
        for j in range(self.n_layers):
            for i in range(N):
                k    = int(gate_idx[j, i])
                name = _GATE_NAMES[k]
                if name == "I":
                    continue
                elif name in _ROT_AXIS:
                    gates.append((j, name, (i,), float(theta_np[j, i])))
                else:   # CNOT
                    ctrl = (i + 1) % N
                    gates.append((j, "CNOT", (ctrl, i), None))
        return gates

    def _build_state_tensor(self, gate_idx: torch.Tensor) -> torch.Tensor:
        """
        Build a PSQASBench-format L × (N+6) × N state tensor from a
        (L, N) gate-index tensor.

        State tensor layout (matching environment.py / pcd.py):
            [layer, 0:N,   :]  — CNOT adjacency:  state[l][targ][ctrl] = 1
            [layer, N:N+3, :]  — rotation indicator: state[l][N+axis][q] = 1
            [layer, N+3:,  :]  — rotation angle:     state[l][N+3+axis][q] = angle
        """
        N         = self.n_qubits
        L         = self.n_layers
        state     = torch.zeros(L, N + 6, N)
        theta_np  = self._circuit.theta.detach().cpu().numpy()
        gate_idx_np = gate_idx.cpu().numpy()

        for j in range(L):
            for i in range(N):
                k    = int(gate_idx_np[j, i])
                name = _GATE_NAMES[k]
                if name == "I":
                    continue
                elif name in _ROT_AXIS:
                    axis = _ROT_AXIS[name]
                    state[j, N + axis,     i] = 1.0
                    state[j, N + 3 + axis, i] = float(theta_np[j, i])
                else:   # CNOT: target = i, control = (i+1) % N
                    ctrl = (i + 1) % N
                    state[j, i, ctrl] = 1.0

        return state

    def _eval_with_gate_idx(self, gate_idx: torch.Tensor) -> dict:
        """
        Evaluate a discrete circuit given its (L, N) gate-index tensor.

        Builds PSQASBench state tensor → qulacs circuit → |ψ⟩ → ⟨ψ|H|ψ⟩.
        Returns the standard rollout-result dict.
        """
        import sys, pathlib
        sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
        from metrics.pcd import state_tensor_to_circuit, circuit_to_statevector

        N            = self.n_qubits
        L            = self.n_layers
        state_tensor = self._build_state_tensor(gate_idx)

        qcirc    = state_tensor_to_circuit(state_tensor, N)
        psi      = circuit_to_statevector(qcirc, N)
        energy   = float(np.real(psi.conj() @ self.hamiltonian @ psi)) + self.energy_shift
        error_ha = abs(energy - self.exact_energy)

        gates      = self._gate_idx_to_list(gate_idx)
        n_cnot     = sum(1 for _, g, _, _ in gates if g == "CNOT")
        n_rot      = sum(1 for _, g, _, _ in gates if g in _ROT_AXIS)
        op_history = self._gates_to_op_history(gates)

        return {
            "energy":         energy,
            "energy_error":   error_ha,
            "cnot_count":     n_cnot,
            "rotation_count": n_rot,
            "success":        int(error_ha < self.accept_err),
            "steps":          L,
            "op_history":     op_history,
            "n_qubits":       N,
            "final_state":    state_tensor,
        }

    # ── Result file writers ───────────────────────────────────────────────────

    def _write_run_meta(self, t_total: float):
        """Write run_meta.txt — mirrors RLQAS ResultLogger.write_run_meta()."""
        path = self.result_dir / "run_meta.txt"
        with open(path, "w") as f:
            f.write(f"method          = QuantumDARTS\n")
            f.write(f"mol_path        = {self.mol_path}\n")
            f.write(f"seed            = {self.seed}\n")
            f.write(f"exact_energy_ha = {self.exact_energy:.8f}\n")
            f.write(f"accept_err_ha   = {self.accept_err:.6f}\n")
            f.write(f"n_qubits        = {self.n_qubits}\n")
            f.write(f"num_layers      = {self.n_layers}\n")
            f.write(f"n_epochs        = {self.n_epochs}\n")
            f.write(f"n_inner         = {self.n_inner}\n")
            f.write(f"K_prime         = {self.K_prime}\n")
            f.write(f"lr_theta        = {self.lr_theta}\n")
            f.write(f"lr_arch         = {self.lr_arch}\n")
            f.write(f"tau_start       = {self.tau_start}\n")
            f.write(f"tau_end         = {self.tau_end}\n")
            f.write(f"finetune_steps  = {self.finetune_steps}\n")
            f.write(f"lr_finetune     = {self.lr_finetune}\n")
            f.write(f"total_time_s    = {t_total:.1f}\n")

    def _write_best_circuit(
        self,
        best_energy: float,
        error_ha: float,
        n_cnot: int,
        gates: list,
    ):
        """Write best_circuit.txt — mirrors RLQAS ResultLogger._write_best_eval()."""
        path = self.result_dir / "best_circuit.txt"
        success = int(error_ha < self.accept_err)
        with open(path, "w") as f:
            f.write(f"energy_ha        = {best_energy:.8f}\n")
            f.write(f"energy_error_ha  = {error_ha:.8f}\n")
            f.write(f"energy_error_mha = {error_ha * 1000:.4f}\n")
            f.write(f"cnot_count       = {n_cnot}\n")
            f.write(f"success          = {success}\n")
            f.write(f"circuit_depth    = {self.n_layers}\n")
            f.write(f"\n--- discrete circuit (layer  qubit → gate) ---\n")
            for layer, name, qubits, angle in gates:
                if name == "CNOT":
                    ctrl, targ = qubits
                    f.write(f"  layer={layer}  CNOT  ctrl={ctrl}  targ={targ}\n")
                else:
                    f.write(f"  layer={layer}  {name:<4s}  q={qubits[0]}  angle={angle:.6f}\n")

    def _write_convergence_history(
        self,
        search_hist: list,
        finetune_hist: list,
    ):
        """Write convergence_history.txt with Phase 1 + Phase 2 energies.

        Format (tab-separated):
            phase  step  energy_ha  energy_error_ha
        """
        path = self.result_dir / "convergence_history.txt"
        with open(path, "w") as f:
            f.write("phase\tstep\tenergy_ha\tenergy_error_ha\n")
            for i, e in enumerate(search_hist):
                f.write(f"search\t{i+1}\t{e:.8f}\t{abs(e - self.exact_energy):.8f}\n")
            for i, e in enumerate(finetune_hist):
                f.write(f"finetune\t{i+1}\t{e:.8f}\t{abs(e - self.exact_energy):.8f}\n")

    def _gates_to_op_history(self, gates: list) -> list:
        """Convert QuantumDARTS gate list to PSQASBench op_history dict format."""
        op_history = []
        for layer, name, qubits, angle in gates:
            if name == "CNOT":
                ctrl, targ = qubits
                op_history.append({"type": "cnot", "layer": int(layer),
                                   "ctrl": int(ctrl), "targ": int(targ)})
            else:
                op_history.append({
                    "type":  "rot",
                    "layer": int(layer),
                    "q":     int(qubits[0]),
                    "axis":  _ROT_AXIS[name] + 1,   # PSQASBench: 1=RX, 2=RY, 3=RZ
                    "angle": float(angle) if angle is not None else 0.0,
                })
        return op_history
