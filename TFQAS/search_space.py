"""
search_space.py — TF-QAS circuit sampling strategies.

The primary layerwise pipeline samples structured half-layers:

  - Single-qubit half-layer: choose one gate type from {rx, ry, rz} and
    place it on either the even or odd qubit subset.
  - Two-qubit half-layer: place gates on either the even-offset or odd-offset
    adjacent pairs.

`n_layers` is the **maximum qubit-level circuit depth**.  Half-layers are
added until the next addition would exceed this limit, giving circuits with
variable gate counts but bounded qubit depth.  Different random offset
choices lead to different qubit-depth profiles and therefore genuinely
different circuit structures.
"""

from __future__ import annotations

import numpy as np

from .circuit import Circuit, SINGLE_GATES, TWO_QUBIT_GATES


def _adjacent_pairs(n_qubits: int, offset: int) -> list[tuple[int, int]]:
    return [(q, q + 1) for q in range(offset, n_qubits - 1, 2)]


def sample_layerwise(
    n_qubits: int,
    n_layers: int,
    rng: np.random.Generator,
    gate_mode: str = "primitive",
) -> Circuit:
    """Sample a layerwise circuit bounded by qubit depth n_layers.

    Half-layers are added until the next half-layer would push the qubit-level
    circuit depth past n_layers.  Different random offset choices create
    different per-qubit depth profiles, so circuits naturally vary in structure
    while staying within the same depth budget.

    gate_mode='primitive' : two-qubit half-layer uses CNOT (non-parameterised).
    gate_mode='paper'     : two-qubit half-layer uses {xx, yy, zz} (parameterised).
    """
    gates: list[tuple[str, tuple[int, ...]]] = []
    moments = [0] * n_qubits  # qubit-level depth for each qubit

    while True:
        # --- single-qubit half-layer ---
        single_gate = str(rng.choice(tuple(SINGLE_GATES)))
        single_offset = int(rng.integers(0, 2))
        rot_qubits = list(range(single_offset, n_qubits, 2))

        new_moments = list(moments)
        for q in rot_qubits:
            new_moments[q] += 1
        if max(new_moments) > n_layers:
            break
        moments = new_moments
        for q in rot_qubits:
            gates.append((single_gate, (q,)))

        # --- two-qubit half-layer ---
        pair_offset = int(rng.integers(0, 2))
        pairs = _adjacent_pairs(n_qubits, pair_offset)

        new_moments = list(moments)
        for q0, q1 in pairs:
            d = max(new_moments[q0], new_moments[q1]) + 1
            new_moments[q0] = d
            new_moments[q1] = d
        if max(new_moments) > n_layers:
            break
        moments = new_moments

        if gate_mode == "primitive":
            for q0, q1 in pairs:
                gates.append(("cnot", (q0, q1)))
        else:
            two_gate = str(rng.choice(tuple(TWO_QUBIT_GATES)))
            for q0, q1 in pairs:
                gates.append((two_gate, (q0, q1)))

    return Circuit(n_qubits, gates)


def sample_random(
    n_qubits: int,
    n_gates: int,
    rng: np.random.Generator,
    two_qubit_prob: float = 0.5,
    connectivity: str = "all",
    gate_mode: str = "primitive",
) -> Circuit:
    gates: list[tuple[str, tuple[int, ...]]] = []
    connectivity = str(connectivity).strip().lower()

    for _ in range(n_gates):
        if n_qubits >= 2 and rng.random() < two_qubit_prob:
            if connectivity == "linear":
                q0 = int(rng.integers(0, n_qubits - 1))
                q1 = q0 + 1
            else:
                q0, q1 = sorted(rng.choice(n_qubits, size=2, replace=False).tolist())
            if gate_mode == "primitive":
                gates.append(("cnot", (q0, q1)))
            else:
                gate_type = str(rng.choice(tuple(TWO_QUBIT_GATES)))
                gates.append((gate_type, (q0, q1)))
        else:
            gate_type = str(rng.choice(tuple(SINGLE_GATES)))
            q = int(rng.integers(0, n_qubits))
            gates.append((gate_type, (q,)))

    return Circuit(n_qubits, gates)
