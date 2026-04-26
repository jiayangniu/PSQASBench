"""
energy_eval.py — PennyLane statevector utilities for GQE.

The benchmark Hamiltonian already exists as a dense matrix inside the molecule
`.npz`, so this module computes energies from saved state snapshots rather than
reconstructing a PennyLane Hamiltonian observable.
"""

from __future__ import annotations

import numpy as np


def build_state_snapshot_circuit(init_state: np.ndarray, num_qubits: int):
    import pennylane as qml

    dev = qml.device("default.qubit", wires=num_qubits)

    @qml.qnode(dev)
    def state_circuit(gqe_ops):
        qml.BasisState(init_state, wires=range(num_qubits))
        for op in gqe_ops:
            qml.Snapshot(measurement=qml.state())
            qml.apply(op)
        return qml.state()

    return qml.snapshots(state_circuit)


def _state_energy(state: np.ndarray, hamiltonian_matrix: np.ndarray) -> float:
    return float(np.real(state.conj() @ (hamiltonian_matrix @ state)))


def get_subsequence_energies(
    op_seq: list[list],
    *,
    snapshot_circuit,
    hamiltonian_matrix: np.ndarray,
    energy_shift: float = 0.0,
    log_interval: int = 200,
) -> np.ndarray:
    """
    Compute energies for every non-empty prefix of each operator sequence.

    Returns shape `(n_sequences, seq_len)` where element `[i, k]` is the energy
    after applying operators `0..k` from sequence `i`.
    """
    rows = []
    total = len(op_seq)
    for idx, ops in enumerate(op_seq):
        snapshots = snapshot_circuit(ops)
        states = [snapshots[k] for k in list(range(1, len(ops))) + ["execution_results"]]
        rows.append(
            [
                _state_energy(np.asarray(state), hamiltonian_matrix) + float(energy_shift)
                for state in states
            ]
        )
        if log_interval and (idx + 1) % log_interval == 0:
            print(f"  [gqe energy] {idx + 1}/{total} sequences done…", flush=True)
    return np.asarray(rows, dtype=np.float64)
