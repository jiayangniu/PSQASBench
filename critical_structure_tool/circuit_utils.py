from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from qulacs import QuantumCircuit as QC, QuantumState

from .types import GateSpec


def gate_dicts_to_gates(gate_dicts: list[dict]) -> list[GateSpec]:
    """Convert a list of direct gate-spec dicts (TFQAS/DARTS format) to GateSpec objects.

    Each dict must have one of the following shapes::

        {"type": "cnot", "ctrl": int, "targ": int}
        {"type": "rot",  "q": int, "axis": int, "angle": float}  # axis: 1=RX 2=RY 3=RZ

    The *step_index* and *action_id* are both set to the gate's position in the list so
    that ``structure_signature`` and ``detailed_signature`` remain stable and unique.
    """
    gates: list[GateSpec] = []
    for i, d in enumerate(gate_dicts):
        gate_type = str(d.get("type", "")).lower()
        if gate_type == "cnot":
            gates.append(
                GateSpec(
                    gate_type="cnot",
                    action_id=i,
                    step_index=i,
                    ctrl=int(d["ctrl"]),
                    targ=int(d["targ"]),
                )
            )
        elif gate_type == "rot":
            gates.append(
                GateSpec(
                    gate_type="rot",
                    action_id=i,
                    step_index=i,
                    q=int(d["q"]),
                    axis=int(d["axis"]),
                    angle=float(d.get("angle", 0.0)),
                )
            )
        else:
            # Unknown gate type (e.g. "ucc_single", "ucc_double" from GQE) —
            # skip silently.  Circuit-level analysis is not applicable for
            # these gates; method-specific analysis should be done separately.
            continue
    return gates


def action_id_to_gate(
    action_id: int,
    step_index: int,
    action_dict: dict[int, list[int]],
    num_qubits: int,
) -> GateSpec:
    ctrl, offset, rot_qubit, rot_axis = action_dict[action_id]
    if rot_qubit < num_qubits:
        return GateSpec(
            gate_type="rot",
            action_id=action_id,
            step_index=step_index,
            q=rot_qubit,
            axis=rot_axis,
            angle=0.0,
        )
    targ = (ctrl + offset) % num_qubits
    return GateSpec(
        gate_type="cnot",
        action_id=action_id,
        step_index=step_index,
        ctrl=ctrl,
        targ=targ,
    )


def apply_snapshot(
    gates: list[GateSpec],
    snapshot: dict | None,
) -> list[GateSpec]:
    if not snapshot:
        return gates

    gate_params = snapshot.get("gate_params", [])
    param_step_indices = snapshot.get("param_step_indices", [])
    if not isinstance(gate_params, list) or not isinstance(param_step_indices, list):
        return gates
    if len(gate_params) != len(param_step_indices):
        return gates

    step_to_angle = {}
    for step_index, angle in zip(param_step_indices, gate_params):
        try:
            step_to_angle[int(step_index)] = float(angle)
        except (TypeError, ValueError):
            continue

    hydrated: list[GateSpec] = []
    for gate in gates:
        new_gate = GateSpec(**gate.__dict__)
        if gate.gate_type == "rot" and gate.step_index in step_to_angle:
            new_gate.angle = step_to_angle[gate.step_index]
        hydrated.append(new_gate)
    return hydrated


def prefix_to_gates(
    action_ids: list[int],
    action_dict: dict[int, list[int]],
    num_qubits: int,
    snapshot: dict | None = None,
    first_hit_snapshot: dict | None = None,
) -> list[GateSpec]:
    gates = [
        action_id_to_gate(action_id, step_index=i, action_dict=action_dict, num_qubits=num_qubits)
        for i, action_id in enumerate(action_ids)
    ]
    chosen_snapshot = snapshot if snapshot is not None else first_hit_snapshot
    return apply_snapshot(gates, chosen_snapshot)


def build_qulacs_circuit(gates: list[GateSpec], n_qubits: int) -> QC:
    qc = QC(n_qubits)
    for gate in gates:
        if gate.gate_type == "cnot":
            qc.add_CNOT_gate(gate.ctrl, gate.targ)
        else:
            if gate.axis == 1:
                qc.add_RX_gate(gate.q, float(gate.angle))
            elif gate.axis == 2:
                qc.add_RY_gate(gate.q, float(gate.angle))
            elif gate.axis == 3:
                qc.add_RZ_gate(gate.q, float(gate.angle))
            else:
                raise ValueError(f"Unsupported rotation axis: {gate.axis}")
    return qc


def eval_energy(gates: list[GateSpec], n_qubits: int, H: np.ndarray, energy_shift: float) -> float:
    if not gates:
        psi = np.zeros(2**n_qubits, dtype=np.complex128)
        psi[0] = 1.0
        return float(np.real(psi.conj() @ H @ psi)) + energy_shift

    qc = build_qulacs_circuit(gates, n_qubits)
    st = QuantumState(n_qubits)
    st.set_zero_state()
    qc.update_quantum_state(st)
    vec = st.get_vector()
    return float(np.real(vec.conj() @ H @ vec)) + energy_shift


def rotation_angles(gates: list[GateSpec]) -> np.ndarray:
    return np.array([gate.angle for gate in gates if gate.gate_type == "rot"], dtype=float)


def clone_with_angles(gates: list[GateSpec], angles: np.ndarray) -> list[GateSpec]:
    cloned: list[GateSpec] = []
    idx = 0
    for gate in gates:
        new_gate = GateSpec(**gate.__dict__)
        if gate.gate_type == "rot":
            new_gate.angle = float(angles[idx])
            idx += 1
        cloned.append(new_gate)
    return cloned


def optimize_circuit(
    gates: list[GateSpec],
    n_qubits: int,
    H: np.ndarray,
    energy_shift: float,
    init_angles: np.ndarray | None = None,
    maxiter: int = 300,
    n_restarts: int = 2,
    optimizer: str = "cobyla",
    rotosolve_sweeps: int = 2,
    rng: np.random.Generator | None = None,
) -> tuple[float, list[GateSpec]]:
    rot_count = sum(g.gate_type == "rot" for g in gates)
    if rot_count == 0:
        energy = eval_energy(gates, n_qubits, H, energy_shift)
        return energy, clone_with_angles(gates, np.array([], dtype=float))

    rng = rng or np.random.default_rng()
    stored = rotation_angles(gates)

    def cost(angles: np.ndarray) -> float:
        return eval_energy(clone_with_angles(gates, angles), n_qubits, H, energy_shift)

    method = str(optimizer).lower()
    if method == "rotosolve":
        probe_vals = np.array([0.0, np.pi / 2.0, np.pi], dtype=float)

        best_energy = float("inf")
        best_angles = stored.copy()

        for restart in range(max(1, n_restarts)):
            if restart == 0 and init_angles is not None and len(init_angles) == rot_count:
                angles = np.array(init_angles, dtype=float)
            elif restart == 0:
                angles = stored.copy()
            else:
                angles = rng.uniform(-np.pi, np.pi, rot_count)

            for _ in range(max(1, int(rotosolve_sweeps))):
                for i in range(rot_count):
                    original = angles[i]
                    e_vals = []
                    for probe in probe_vals:
                        angles[i] = probe
                        e_vals.append(cost(angles))
                    angles[i] = original
                    e0, epi2, epi = e_vals
                    A = (e0 - epi) / 2.0
                    C = (e0 + epi) / 2.0
                    B = epi2 - C
                    angles[i] = float(np.arctan2(-B, -A))

            energy = cost(angles)
            if energy < best_energy:
                best_energy = energy
                best_angles = angles.copy()

        return float(best_energy), clone_with_angles(gates, best_angles)

    best_energy = float("inf")
    best_angles = stored.copy()

    for restart in range(max(1, n_restarts)):
        if restart == 0 and init_angles is not None and len(init_angles) == rot_count:
            x0 = init_angles.copy()
        elif restart == 0:
            x0 = stored.copy()
        else:
            x0 = rng.uniform(-np.pi, np.pi, rot_count)

        res = minimize(cost, x0, method="COBYLA", options={"maxiter": maxiter, "rhobeg": 1.0})
        if res.fun < best_energy:
            best_energy = float(res.fun)
            best_angles = np.array(res.x, dtype=float)

    return best_energy, clone_with_angles(gates, best_angles)


def remove_gate(gates: list[GateSpec], gate_index: int) -> tuple[list[GateSpec], np.ndarray]:
    init_angles = []
    remaining = []
    for idx, gate in enumerate(gates):
        if idx == gate_index:
            continue
        remaining.append(GateSpec(**gate.__dict__))
        if gate.gate_type == "rot":
            init_angles.append(float(gate.angle))
    return remaining, np.array(init_angles, dtype=float)
