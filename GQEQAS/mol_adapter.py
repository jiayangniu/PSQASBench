"""
mol_adapter.py — PSQASBench .npz  →  PennyLane objects for GQE training.

The GQE method (Nakaji et al. 2024) needs three PennyLane objects that are
not stored directly in PSQASBench's .npz files:
  1. qml.Hamiltonian  — built by decomposing the dense matrix into Pauli strings
  2. hf_state array   — first active_electrons spin-orbitals occupied
  3. op_pool list     — UCC single/double excitation operators with 8 time values

The only extra piece of information required beyond the .npz is
``active_electrons`` (the number of electrons in the active space), which must
be supplied via the config file.  ``active_orbitals`` defaults to
``n_qubits // 2``, which is correct for Jordan-Wigner encoding.

Warning: ``qml.pauli_decompose`` scales as O(4^n), so for n > 10 qubits this
step can take several minutes.  Consider pre-computing and caching for large
molecules.
"""
import numpy as np
import pennylane as qml
from pennylane import qchem

# 8 logarithmically-spaced time parameters matching Q-GESolver's default pool
_OP_TIMES = np.sort(np.logspace(-2, 0, num=8)) / 160


def build_pennylane_objects(
    npz_path,
    active_electrons: int,
    active_orbitals: int | None = None,
) -> dict:
    """Return PennyLane objects built from a PSQASBench .npz file.

    Parameters
    ----------
    npz_path : path-like
        Path to the PSQASBench .npz Hamiltonian file.
    active_electrons : int
        Number of electrons in the active space (must match the molecule).
    active_orbitals : int, optional
        Number of spatial orbitals in the active space.
        Defaults to ``n_qubits // 2`` (correct for Jordan-Wigner encoding).

    Returns
    -------
    dict with keys:
        hamiltonian   : qml.Hamiltonian
        hf_state      : np.ndarray, shape (n_qubits,), dtype int
        op_pool       : list[qml.Operation]
        exact_energy  : float (Ha, includes nuclear repulsion shift)
        energy_shift  : float (Ha, nuclear repulsion)
        n_qubits      : int
        active_electrons : int
        active_orbitals  : int
    """
    data = np.load(str(npz_path), allow_pickle=True)
    H_matrix     = data["hamiltonian"]
    energy_shift = float(data.get("energy_shift", 0.0))
    exact_energy = float(np.min(data["eigvals"])) + energy_shift
    n_qubits     = int(round(np.log2(H_matrix.shape[0])))

    if active_orbitals is None:
        active_orbitals = n_qubits // 2

    # Dense matrix → PennyLane Hamiltonian via Pauli decomposition.
    # This is O(4^n); for n_qubits > 10 a progress message is shown.
    if n_qubits > 10:
        print(f"  [mol_adapter] pauli_decompose for {n_qubits}q "
              f"({2**n_qubits}×{2**n_qubits} matrix) — this may take a few minutes …",
              flush=True)
    H_pl = qml.pauli_decompose(H_matrix, wire_order=range(n_qubits))

    # HF state: first active_electrons spin-orbitals occupied, rest empty.
    hf_state = np.array(
        [1] * active_electrons + [0] * (n_qubits - active_electrons), dtype=int
    )

    # UCC operator pool: single and double excitations × 8 time values.
    singles, doubles = qchem.excitations(active_electrons, n_qubits)
    op_pool: list[qml.Operation] = []
    for s in singles:
        for t in _OP_TIMES:
            op_pool.append(qml.SingleExcitation(float(t), wires=s))
            op_pool.append(qml.SingleExcitationMinus(float(t), wires=s))
    for d in doubles:
        for t in _OP_TIMES:
            op_pool.append(qml.DoubleExcitation(float(t), wires=d))
            op_pool.append(qml.DoubleExcitationMinus(float(t), wires=d))

    return {
        "hamiltonian":      H_pl,
        "hf_state":         hf_state,
        "op_pool":          op_pool,
        "exact_energy":     exact_energy,
        "energy_shift":     energy_shift,
        "n_qubits":         n_qubits,
        "active_electrons": active_electrons,
        "active_orbitals":  active_orbitals,
    }
