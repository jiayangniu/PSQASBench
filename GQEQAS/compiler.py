"""
compiler.py — convert GQE operator sequences to benchmark primitive gates.
"""

from __future__ import annotations

import math


_AXIS_MAP = {"RX": 1, "RY": 2, "RZ": 3}


def compile_ops_to_primitives(ops) -> list[dict]:
    compiled: list[dict] = []
    for token_index, op in enumerate(ops):
        meta = {
            "source_token": int(token_index),
            "source_name": str(getattr(op, "name", type(op).__name__)),
            "source_wires": [int(w) for w in getattr(op, "wires", [])],
            "source_parameters": [float(p) for p in getattr(op, "parameters", ())],
        }
        _lower_op(op, compiled, meta)
    for layer, gate in enumerate(compiled):
        gate["layer"] = int(layer)
    return compiled


def _append_rot(out: list[dict], *, q: int, axis: int, angle: float, meta: dict) -> None:
    gate = {"type": "rot", "q": int(q), "axis": int(axis), "angle": float(angle)}
    gate.update(meta)
    out.append(gate)


def _append_cnot(out: list[dict], *, ctrl: int, targ: int, meta: dict) -> None:
    gate = {"type": "cnot", "ctrl": int(ctrl), "targ": int(targ)}
    gate.update(meta)
    out.append(gate)


def _append_z_string_phase(
    out: list[dict],
    *,
    wires: list[int],
    angle: float,
    meta: dict,
) -> None:
    """Append exp(-i * angle * Z_{w0} ... Z_{wk}) using RZ+CNOT parity gadgets."""
    if not wires:
        return
    if len(wires) == 1:
        _append_rot(out, q=wires[0], axis=3, angle=2.0 * float(angle), meta=meta)
        return
    target = int(wires[-1])
    ctrls = [int(w) for w in wires[:-1]]
    for ctrl in ctrls:
        _append_cnot(out, ctrl=ctrl, targ=target, meta=meta)
    _append_rot(out, q=target, axis=3, angle=2.0 * float(angle), meta=meta)
    for ctrl in reversed(ctrls):
        _append_cnot(out, ctrl=ctrl, targ=target, meta=meta)


def _lower_single_excitation(phi: float, wires: list[int], out: list[dict], meta: dict) -> None:
    q0, q1 = [int(w) for w in wires]
    _lower_op(type("_Op", (), {"name": "Hadamard", "wires": [q0], "parameters": ()})(), out, meta)
    _append_cnot(out, ctrl=q0, targ=q1, meta=meta)
    _append_rot(out, q=q0, axis=2, angle=-phi / 2.0, meta=meta)
    _append_rot(out, q=q1, axis=2, angle=-phi / 2.0, meta=meta)
    _append_cnot(out, ctrl=q0, targ=q1, meta=meta)
    _lower_op(type("_Op", (), {"name": "Hadamard", "wires": [q0], "parameters": ()})(), out, meta)


def _lower_double_excitation(phi: float, wires: list[int], out: list[dict], meta: dict) -> None:
    w0, w1, w2, w3 = [int(w) for w in wires]
    _append_cnot(out, ctrl=w2, targ=w3, meta=meta)
    _append_cnot(out, ctrl=w0, targ=w2, meta=meta)
    _lower_op(type("_Op", (), {"name": "Hadamard", "wires": [w3], "parameters": ()})(), out, meta)
    _lower_op(type("_Op", (), {"name": "Hadamard", "wires": [w0], "parameters": ()})(), out, meta)
    _append_cnot(out, ctrl=w2, targ=w3, meta=meta)
    _append_cnot(out, ctrl=w0, targ=w1, meta=meta)
    _append_rot(out, q=w1, axis=2, angle=phi / 8.0, meta=meta)
    _append_rot(out, q=w0, axis=2, angle=-phi / 8.0, meta=meta)
    _append_cnot(out, ctrl=w0, targ=w3, meta=meta)
    _lower_op(type("_Op", (), {"name": "Hadamard", "wires": [w3], "parameters": ()})(), out, meta)
    _append_cnot(out, ctrl=w3, targ=w1, meta=meta)
    _append_rot(out, q=w1, axis=2, angle=phi / 8.0, meta=meta)
    _append_rot(out, q=w0, axis=2, angle=-phi / 8.0, meta=meta)
    _append_cnot(out, ctrl=w2, targ=w1, meta=meta)
    _append_cnot(out, ctrl=w2, targ=w0, meta=meta)
    _append_rot(out, q=w1, axis=2, angle=-phi / 8.0, meta=meta)
    _append_rot(out, q=w0, axis=2, angle=phi / 8.0, meta=meta)
    _append_cnot(out, ctrl=w3, targ=w1, meta=meta)
    _lower_op(type("_Op", (), {"name": "Hadamard", "wires": [w3], "parameters": ()})(), out, meta)
    _append_cnot(out, ctrl=w0, targ=w3, meta=meta)
    _append_rot(out, q=w1, axis=2, angle=-phi / 8.0, meta=meta)
    _append_rot(out, q=w0, axis=2, angle=phi / 8.0, meta=meta)
    _append_cnot(out, ctrl=w0, targ=w1, meta=meta)
    _append_cnot(out, ctrl=w2, targ=w0, meta=meta)
    _lower_op(type("_Op", (), {"name": "Hadamard", "wires": [w0], "parameters": ()})(), out, meta)
    _lower_op(type("_Op", (), {"name": "Hadamard", "wires": [w3], "parameters": ()})(), out, meta)
    _append_cnot(out, ctrl=w0, targ=w2, meta=meta)
    _append_cnot(out, ctrl=w2, targ=w3, meta=meta)


def _adjoint_base_name(name: str) -> str | None:
    if name.startswith("Adjoint(") and name.endswith(")"):
        return name[len("Adjoint(") : -1]
    return None


def _lower_known_leaf(name: str, wires: list[int], params: list[float], out: list[dict], meta: dict) -> bool:
    if name in _AXIS_MAP:
        _append_rot(out, q=wires[0], axis=_AXIS_MAP[name], angle=params[0], meta=meta)
        return True
    if name == "Rot":
        phi, theta, omega = params
        _append_rot(out, q=wires[0], axis=3, angle=phi, meta=meta)
        _append_rot(out, q=wires[0], axis=2, angle=theta, meta=meta)
        _append_rot(out, q=wires[0], axis=3, angle=omega, meta=meta)
        return True
    if name == "CNOT":
        _append_cnot(out, ctrl=wires[0], targ=wires[1], meta=meta)
        return True
    if name in {"Identity", "GlobalPhase"}:
        return True
    if name in {"PauliX", "X"}:
        _append_rot(out, q=wires[0], axis=1, angle=math.pi, meta=meta)
        return True
    if name in {"PauliY", "Y"}:
        _append_rot(out, q=wires[0], axis=2, angle=math.pi, meta=meta)
        return True
    if name in {"PauliZ", "Z"}:
        _append_rot(out, q=wires[0], axis=3, angle=math.pi, meta=meta)
        return True
    if name == "Hadamard":
        _append_rot(out, q=wires[0], axis=3, angle=math.pi, meta=meta)
        _append_rot(out, q=wires[0], axis=2, angle=math.pi / 2.0, meta=meta)
        return True
    if name == "S":
        _append_rot(out, q=wires[0], axis=3, angle=math.pi / 2.0, meta=meta)
        return True
    if name == "T":
        _append_rot(out, q=wires[0], axis=3, angle=math.pi / 4.0, meta=meta)
        return True
    if name == "PhaseShift":
        _append_rot(out, q=wires[0], axis=3, angle=params[0], meta=meta)
        return True
    base = _adjoint_base_name(name)
    if base == "S":
        _append_rot(out, q=wires[0], axis=3, angle=-math.pi / 2.0, meta=meta)
        return True
    if base == "T":
        _append_rot(out, q=wires[0], axis=3, angle=-math.pi / 4.0, meta=meta)
        return True
    if base in _AXIS_MAP:
        _append_rot(out, q=wires[0], axis=_AXIS_MAP[base], angle=-params[0], meta=meta)
        return True
    return False


def _lower_op(op, out: list[dict], meta: dict) -> None:
    name = str(getattr(op, "name", type(op).__name__))
    wires = [int(w) for w in getattr(op, "wires", [])]
    params = [float(p) for p in getattr(op, "parameters", ())]

    if name == "SingleExcitation":
        _lower_single_excitation(params[0], wires, out, meta)
        return

    if name == "SingleExcitationMinus":
        phi = params[0]
        _lower_single_excitation(phi, wires, out, meta)
        # Up to a global phase, this is exp(-i * phi/4 * Z_i Z_j) times SingleExcitation.
        _append_z_string_phase(out, wires=wires, angle=phi / 4.0, meta=meta)
        return

    if name == "DoubleExcitation":
        _lower_double_excitation(params[0], wires, out, meta)
        return

    if name == "DoubleExcitationMinus":
        phi = params[0]
        w0, w1, w2, w3 = wires
        _lower_double_excitation(phi, wires, out, meta)
        # Up to a global phase, this applies a diagonal phase correction that is
        # constant on the active |0011>, |1100> rotation subspace and equal to
        # e^{-i phi/2} on the complement.
        phase_terms = (
            ([w0, w1], -phi / 16.0),
            ([w0, w2],  phi / 16.0),
            ([w1, w2],  phi / 16.0),
            ([w0, w3],  phi / 16.0),
            ([w1, w3],  phi / 16.0),
            ([w2, w3], -phi / 16.0),
            ([w0, w1, w2, w3], -phi / 16.0),
        )
        for phase_wires, phase_angle in phase_terms:
            _append_z_string_phase(out, wires=phase_wires, angle=phase_angle, meta=meta)
        return

    if _lower_known_leaf(name, wires, params, out, meta):
        return

    decomp = None
    if hasattr(op, "decomposition"):
        try:
            decomp = op.decomposition()
        except Exception:
            decomp = None
    if decomp:
        for sub_op in decomp:
            _lower_op(sub_op, out, meta)
        return

    raise ValueError(
        f"Unsupported PennyLane operator during GQE export: {name} "
        f"(wires={wires}, params={params})"
    )
