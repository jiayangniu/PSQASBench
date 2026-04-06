"""
Policy Circuit Diversity (PCD) metrics for PSQASBench.

Two complementary metrics, both based on Hilbert-Schmidt (HS) distance:

    D_struct  Structural diversity of the CNOT skeleton.
              Rotation angles are fixed to π/4 before computing the unitary,
              so only the gate topology contributes.

    D_func    Functional diversity of the optimised circuits.
              Each rollout's final COBYLA-optimised angles are used directly
              (no re-optimisation by default; see `reoptimize` flag).

Diagnostic matrix:
    D_struct low  + D_func low   → ideal: consistent structure, stable optimisation
    D_struct high + D_func low   → acceptable: equivalent circuits (Hamiltonian symmetry)
    D_struct low  + D_func high  → landscape problem: structure consistent but optimisation unstable
    D_struct high + D_func high  → unreliable: random walk

This module has no dependency on any agent or runner — it only uses qulacs and numpy.
"""

from __future__ import annotations

import math
import numpy as np
import torch
from itertools import combinations

from qulacs import QuantumState, QuantumCircuit
from qulacs.gate import CNOT, RX, RY, RZ
import scipy.optimize


# ── Unitary extraction ─────────────────────────────────────────────────────────

def state_tensor_to_circuit(state: torch.Tensor, n_qubits: int,
                             fixed_angle: float | None = None) -> QuantumCircuit:
    """
    Build a qulacs QuantumCircuit from a PSQASBench state tensor.

    Args:
        state       : L × (N+6) × N tensor (channels: CNOT | rot-type | rot-angle)
        n_qubits    : N
        fixed_angle : if not None, override all rotation angles with this value
                      (use math.pi / 4 for D_struct computation).

    Returns:
        A qulacs QuantumCircuit (non-parametric, angles baked in).
    """
    circuit    = QuantumCircuit(n_qubits)
    state_np   = state.detach().cpu().numpy()
    num_layers = state_np.shape[0]

    for layer in range(num_layers):
        # ── CNOT gates ────────────────────────────────────────────────────────
        cnot_pos = np.argwhere(state_np[layer, :n_qubits, :] == 1)
        for targ, ctrl in cnot_pos:           # state layout: [targ][ctrl] = 1
            circuit.add_gate(CNOT(int(ctrl), int(targ)))

        # ── Rotation gates ────────────────────────────────────────────────────
        rot_pos = np.argwhere(state_np[layer, n_qubits:n_qubits + 3, :] == 1)
        for axis, qubit in rot_pos:           # axis: 0=RX, 1=RY, 2=RZ
            if fixed_angle is not None:
                angle = fixed_angle
            else:
                angle = float(state_np[layer, n_qubits + 3 + axis, qubit])

            gate_fn = [RX, RY, RZ][axis]
            circuit.add_gate(gate_fn(int(qubit), angle))

    return circuit


def circuit_to_unitary(circuit: QuantumCircuit, n_qubits: int) -> np.ndarray:
    """
    Extract the 2^n × 2^n unitary matrix of a qulacs QuantumCircuit.

    For n ≤ 8 this is fast; for n = 10 (L6) it is still feasible when K is small.
    """
    d = 1 << n_qubits        # 2^n_qubits
    U = np.empty((d, d), dtype=complex)
    for col in range(d):
        ket = QuantumState(n_qubits)
        ket.set_computational_basis(col)
        circuit.update_quantum_state(ket)
        U[:, col] = ket.get_vector()
    return U


# ── HS distance ───────────────────────────────────────────────────────────────

def hs_distance(U1: np.ndarray, U2: np.ndarray) -> float:
    """
    Hilbert-Schmidt distance between two unitaries:
        d_HS(U1, U2) = 1 - |Tr(U1† U2)|² / d²
    Returns a value in [0, 1].
    """
    d = U1.shape[0]
    overlap = np.trace(U1.conj().T @ U2)
    return float(1.0 - (abs(overlap) ** 2) / (d * d))


# ── D_func re-optimisation (optional) ─────────────────────────────────────────

def _reoptimize_angles(state: torch.Tensor, n_qubits: int,
                       hamiltonian: np.ndarray, weights: np.ndarray,
                       energy_shift: float, maxiter: int = 200) -> torch.Tensor:
    """
    Run a single global COBYLA optimisation over all rotation angles in `state`.
    Returns a new state tensor with the optimised angles.

    This gives a cleaner D_func measurement than using per-step COBYLA results,
    but is more expensive (one COBYLA call per rollout).
    """
    from qulacs import QuantumState as QS

    state_out = state.clone()
    state_np  = state_out.detach().cpu().numpy()
    num_layers = state_np.shape[0]

    # Collect (layer, axis, qubit) positions of rotation gates
    rot_positions = []
    for layer in range(num_layers):
        rot_pos = np.argwhere(state_np[layer, n_qubits:n_qubits + 3, :] == 1)
        for axis, qubit in rot_pos:
            rot_positions.append((layer, int(axis), int(qubit)))

    if not rot_positions:
        return state_out  # no rotations → unitary is already fixed

    x0 = np.array([float(state_np[l, n_qubits + 3 + ax, q])
                   for l, ax, q in rot_positions])

    def _energy(x):
        s = state_np.copy()
        for i, (l, ax, q) in enumerate(rot_positions):
            s[l, n_qubits + 3 + ax, q] = x[i]
        circ = state_tensor_to_circuit(torch.tensor(s, dtype=torch.float32), n_qubits)
        ket  = QS(n_qubits)
        circ.update_quantum_state(ket)
        psi = ket.get_vector()
        return float(np.real(psi.conj() @ hamiltonian @ psi)) + energy_shift

    result = scipy.optimize.minimize(_energy, x0=x0, method="COBYLA",
                                     options={"maxiter": maxiter})

    for i, (l, ax, q) in enumerate(rot_positions):
        state_out[l, n_qubits + 3 + ax, q] = float(result.x[i])

    return state_out


# ── Main PCD computation ───────────────────────────────────────────────────────

def compute_pcd(
    rollout_results: list[dict],
    hamiltonian: np.ndarray,
    weights: np.ndarray,
    energy_shift: float,
    reoptimize: bool = False,
    reoptimize_maxiter: int = 200,
) -> dict:
    """
    Compute D_struct and D_func from K greedy rollout results.

    Args:
        rollout_results   : list of dicts from greedy_rollout_k (must contain
                            'final_state' and 'n_qubits' keys).
        hamiltonian       : 2^n × 2^n Hamiltonian matrix (numpy).
        weights           : Pauli weights (used only if reoptimize=True).
        energy_shift      : nuclear repulsion shift.
        reoptimize        : if True, run a final global COBYLA optimisation on
                            each circuit before computing D_func unitaries.
                            More accurate but K× more expensive.
        reoptimize_maxiter: COBYLA maxiter for the optional re-optimisation.

    Returns:
        dict with keys:
            D_struct  float  pairwise-mean HS distance with θ = π/4
            D_func    float  pairwise-mean HS distance with optimised θ*
            n_pairs   int    number of unitary pairs compared (K choose 2)
    """
    K = len(rollout_results)
    if K < 2:
        return {"D_struct": float("nan"), "D_func": float("nan"), "n_pairs": 0}

    n_qubits = rollout_results[0]["n_qubits"]

    # ── Build unitaries ───────────────────────────────────────────────────────
    U_struct = []   # fixed θ = π/4
    U_func   = []   # COBYLA-optimised θ*

    for r in rollout_results:
        state = r["final_state"]        # L × (N+6) × N tensor

        # D_struct: bake in π/4 for all rotation angles
        circ_s = state_tensor_to_circuit(state, n_qubits, fixed_angle=math.pi / 4)
        U_struct.append(circuit_to_unitary(circ_s, n_qubits))

        # D_func: use existing angles (or re-optimise if requested)
        if reoptimize:
            state_opt = _reoptimize_angles(
                state, n_qubits, hamiltonian, weights, energy_shift, reoptimize_maxiter
            )
        else:
            state_opt = state
        circ_f = state_tensor_to_circuit(state_opt, n_qubits)
        U_func.append(circuit_to_unitary(circ_f, n_qubits))

    # ── Pairwise HS distances ─────────────────────────────────────────────────
    pairs = list(combinations(range(K), 2))

    d_struct_vals = [hs_distance(U_struct[i], U_struct[j]) for i, j in pairs]
    d_func_vals   = [hs_distance(U_func[i],   U_func[j])   for i, j in pairs]

    return {
        "D_struct": float(np.mean(d_struct_vals)),
        "D_func":   float(np.mean(d_func_vals)),
        "n_pairs":  len(pairs),
    }
