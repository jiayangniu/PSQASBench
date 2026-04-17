"""
search_space.py — Circuit sampling strategies.

Layerwise pipeline (paper's primary search space):
    Alternate between:
      • Single-qubit sublayer: one randomly chosen rotation gate per qubit
      • Two-qubit sublayer: CNOT gates on alternating pairs
                            even offset: (0,1),(2,3),…
                            odd offset:  (1,2),(3,4),…

Each call to sample_layerwise() returns one Circuit with the gate list
determined randomly (gate types chosen independently per gate).
"""

import numpy as np
from .circuit import Circuit

SINGLE_GATES = ['rx', 'ry', 'rz']
TWO_QUBIT_GATE = 'cnot'


def sample_layerwise(
    n_qubits: int,
    n_layers: int,
    rng: np.random.Generator,
    cnot_prob: float = 0.7,
) -> Circuit:
    """
    Sample a layerwise circuit.

    Each "layer" consists of:
      1. A single-qubit sublayer: one rotation gate per qubit (type drawn uniformly
         from {rx, ry, rz}).
      2. A two-qubit sublayer: each CNOT position (even or odd qubit pairs,
         alternating per layer) is independently included with probability
         cnot_prob. Skipped positions are replaced by a rotation gate on the
         lower qubit. This randomises circuit topology and creates variation
         in DAG path counts for Stage 1 filtering.

    Args:
        n_qubits:  number of qubits
        n_layers:  number of (single + two-qubit) layer pairs
        rng:       numpy random generator
        cnot_prob: probability of placing a CNOT at each candidate position

    Returns:
        Circuit object
    """
    gates = []

    for layer in range(n_layers):
        # ── Single-qubit sublayer ─────────────────────────────────────────────
        for q in range(n_qubits):
            gate_type = SINGLE_GATES[rng.integers(0, len(SINGLE_GATES))]
            gates.append((gate_type, (q,)))

        # ── Two-qubit sublayer (alternating even/odd offset) ──────────────────
        offset = layer % 2           # 0 → (0,1),(2,3),…   1 → (1,2),(3,4),…
        for q in range(offset, n_qubits - 1, 2):
            if rng.random() < cnot_prob:
                gates.append((TWO_QUBIT_GATE, (q, q + 1)))
            else:
                # Replace with a single-qubit gate to keep circuit length similar
                gate_type = SINGLE_GATES[rng.integers(0, len(SINGLE_GATES))]
                gates.append((gate_type, (q,)))

    return Circuit(n_qubits, gates)


def sample_random(
    n_qubits: int,
    n_gates: int,
    rng: np.random.Generator,
    two_qubit_prob: float = 0.3,
) -> Circuit:
    """
    Sample a random circuit by independently drawing each gate.

    Args:
        n_qubits:       number of qubits
        n_gates:        total number of gates
        rng:            numpy random generator
        two_qubit_prob: probability that each gate is a CNOT (vs a rotation)

    Returns:
        Circuit object
    """
    gates = []
    for _ in range(n_gates):
        if n_qubits >= 2 and rng.random() < two_qubit_prob:
            ctrl = int(rng.integers(0, n_qubits))
            targ = int(rng.integers(0, n_qubits - 1))
            if targ >= ctrl:
                targ += 1
            gates.append((TWO_QUBIT_GATE, (ctrl, targ)))
        else:
            gate_type = SINGLE_GATES[rng.integers(0, len(SINGLE_GATES))]
            q = int(rng.integers(0, n_qubits))
            gates.append((gate_type, (q,)))

    return Circuit(n_qubits, gates)
