"""
mol_adapter.py — benchmark molecule adapter for GQE.

This adapter avoids the expensive dense-matrix -> Pauli decomposition path used
by the earlier prototype.  For GQE training we only need:

  - the dense Hamiltonian matrix from the benchmark .npz
  - the exact ground-state energy
  - a Hartree-Fock occupation bitstring
  - an operator pool compatible with PennyLane execution
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class OperatorPoolConfig:
    pool_kind: str = "ucc"
    include_minus: bool = True
    num_op_times: int = 8
    op_time_min_exp: float = -2.0
    op_time_max_exp: float = 0.0
    op_time_divisor: float = 160.0
    connectivity: str = "all"

    def build_times(self) -> np.ndarray:
        if self.pool_kind == "primitive":
            return np.linspace(np.pi / self.num_op_times, np.pi, self.num_op_times)
        return np.sort(
            np.logspace(self.op_time_min_exp, self.op_time_max_exp, num=self.num_op_times)
        ) / float(self.op_time_divisor)


def _read_scalar(data, key: str, default=None, cast=int):
    if key not in data:
        return default
    value = data[key]
    try:
        return cast(np.asarray(value).item())
    except Exception:
        return default


def _primitive_cnot_pairs(num_qubits: int, connectivity: str) -> list[tuple[int, int]]:
    connectivity = str(connectivity).strip().lower()
    if connectivity == "linear":
        pairs = []
        for ctrl in range(num_qubits - 1):
            pairs.append((ctrl, ctrl + 1))
        for ctrl in range(1, num_qubits):
            pairs.append((ctrl, ctrl - 1))
        return pairs
    return [(ctrl, targ) for ctrl in range(num_qubits) for targ in range(num_qubits) if ctrl != targ]


def _build_primitive_pool(qml, *, n_qubits: int, op_times: np.ndarray, include_minus: bool, connectivity: str):
    op_pool: list = []
    for q in range(n_qubits):
        for angle in op_times:
            theta = float(angle)
            op_pool.append(qml.RX(theta, wires=q))
            op_pool.append(qml.RY(theta, wires=q))
            op_pool.append(qml.RZ(theta, wires=q))
            if include_minus:
                op_pool.append(qml.RX(-theta, wires=q))
                op_pool.append(qml.RY(-theta, wires=q))
                op_pool.append(qml.RZ(-theta, wires=q))
    for ctrl, targ in _primitive_cnot_pairs(n_qubits, connectivity):
        op_pool.append(qml.CNOT(wires=[int(ctrl), int(targ)]))
    return op_pool


def build_gqe_molecule(
    npz_path: str | Path,
    *,
    active_electrons: int,
    active_orbitals: int | None = None,
    pool_cfg: OperatorPoolConfig | None = None,
) -> dict:
    import pennylane as qml
    from pennylane import qchem

    path = Path(npz_path)
    data = np.load(str(path), allow_pickle=True)

    h_matrix = np.asarray(data["hamiltonian"], dtype=np.complex128)
    exact_energy = float(np.min(np.asarray(data["eigvals"], dtype=np.float64)))
    energy_shift = float(data.get("energy_shift", 0.0))
    n_qubits = int(_read_scalar(data, "n_qubits", default=round(np.log2(h_matrix.shape[0]))))

    file_active_e = _read_scalar(data, "active_electrons", default=None)
    file_active_o = _read_scalar(data, "active_orbitals", default=None)

    if active_electrons is None:
        if file_active_e is None:
            raise ValueError("active_electrons must be provided for this molecule.")
        active_electrons = int(file_active_e)
    if active_orbitals is None:
        active_orbitals = int(file_active_o) if file_active_o is not None else n_qubits // 2

    hf_state = qchem.hf_state(int(active_electrons), int(n_qubits))
    pool_cfg = pool_cfg or OperatorPoolConfig()
    op_times = pool_cfg.build_times()

    pool_kind = pool_cfg.pool_kind.lower()
    if pool_kind == "ucc":
        singles, doubles = qchem.excitations(int(active_electrons), int(n_qubits))
        op_pool: list = []
        for single in singles:
            for time in op_times:
                op_pool.append(qml.SingleExcitation(float(time), wires=single))
                if pool_cfg.include_minus:
                    op_pool.append(qml.SingleExcitationMinus(float(time), wires=single))
        for double in doubles:
            for time in op_times:
                op_pool.append(qml.DoubleExcitation(float(time), wires=double))
                if pool_cfg.include_minus:
                    op_pool.append(qml.DoubleExcitationMinus(float(time), wires=double))
    elif pool_kind == "primitive":
        op_pool = _build_primitive_pool(
            qml,
            n_qubits=int(n_qubits),
            op_times=op_times,
            include_minus=bool(pool_cfg.include_minus),
            connectivity=pool_cfg.connectivity,
        )
    else:
        raise ValueError(f"Unsupported GQE operator pool kind '{pool_cfg.pool_kind}'.")

    return {
        "hamiltonian_matrix": h_matrix,
        "hf_state": np.asarray(hf_state, dtype=int),
        "op_pool": op_pool,
        "exact_energy": float(exact_energy + energy_shift),
        "energy_shift": float(energy_shift),
        "n_qubits": int(n_qubits),
        "active_electrons": int(active_electrons),
        "active_orbitals": int(active_orbitals),
        "pool_kind": pool_kind,
        "include_minus": bool(pool_cfg.include_minus),
        "connectivity": str(pool_cfg.connectivity).strip().lower(),
        "num_op_times": int(pool_cfg.num_op_times),
        "op_times": np.asarray(op_times, dtype=np.float64),
        "source_npz": str(path),
    }
