"""
circuit.py — TF-QAS native circuit template.

Supported gate sets:
  paper     : {'rx', 'ry', 'rz', 'xx', 'yy', 'zz'}  (all parameterised)
  primitive : {'rx', 'ry', 'rz', 'cnot'}  (cnot is non-parameterised)

The TF-QAS algorithm operates on the native representation for path counting
and expressibility evaluation.  In 'paper' mode the native gates are later
compiled to primitive {RX, RY, RZ, CNOT} for benchmarking; in 'primitive' mode
the native gates are already primitive so compilation is an identity step.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
from qulacs import QuantumCircuit as QCircuit, QuantumState

SINGLE_GATES = frozenset(["rx", "ry", "rz"])
TWO_QUBIT_GATES = frozenset(["xx", "yy", "zz"])
CNOT_GATES = frozenset(["cnot"])          # non-parameterised two-qubit gate
PARAM_GATES = SINGLE_GATES | TWO_QUBIT_GATES
ALL_GATES = PARAM_GATES | CNOT_GATES

_NATIVE_AXIS = {"rx": 1, "ry": 2, "rz": 3}


class Circuit:
    """
    Quantum circuit template defined as an ordered list of native gates.

    Each gate is represented as ``(gate_type, qubits)``, where ``qubits`` is a
    length-1 tuple for single-qubit gates and a length-2 tuple for two-qubit
    gates.  Parameters are provided externally.
    """

    def __init__(self, n_qubits: int, gates: list[tuple[str, tuple[int, ...]]]):
        self.n_qubits = int(n_qubits)
        self.gates = list(gates)
        # Only parameterised gates consume entries in the parameter vector.
        self.n_params = sum(1 for gate_type, _ in self.gates if gate_type in PARAM_GATES)
        self.n_two_qubit = sum(
            1 for gate_type, _ in self.gates if gate_type in (TWO_QUBIT_GATES | CNOT_GATES)
        )
        # Report compiled CNOT count (benchmark-facing entangling-gate cost).
        # xx/yy/zz each compile to 2 CNOTs; a native cnot counts as 1.
        self.n_cnot = sum(
            2 if gate_type in TWO_QUBIT_GATES else (1 if gate_type in CNOT_GATES else 0)
            for gate_type, _ in self.gates
        )
        self._path_count: int | None = None

    @property
    def path_count(self) -> int:
        if self._path_count is None:
            self._path_count = self._count_dag_paths()
        return self._path_count

    def _count_dag_paths(self) -> int:
        n = len(self.gates)
        if n == 0:
            return 1

        last = [-1] * self.n_qubits
        adj: dict[int, list[int]] = defaultdict(list)

        for i, (_, qubits) in enumerate(self.gates):
            for q in qubits:
                adj[last[q]].append(i)
            for q in qubits:
                last[q] = i

        sink = n
        for q in range(self.n_qubits):
            adj[last[q]].append(sink)

        dp: dict[int, int] = defaultdict(int)
        dp[-1] = 1
        for i in range(-1, n):
            for j in adj[i]:
                dp[j] += dp[i]
        return dp[sink]

    def initial_params(
        self,
        rng: np.random.Generator,
        *,
        low: float = -2.0 * np.pi,
        high: float = 2.0 * np.pi,
    ) -> np.ndarray:
        if self.n_params == 0:
            return np.array([], dtype=np.float64)
        return rng.uniform(low, high, self.n_params).astype(np.float64)

    def statevector(self, params: np.ndarray) -> np.ndarray:
        state = QuantumState(self.n_qubits)
        state.set_zero_state()
        qc = self._build_qulacs(params)
        qc.update_quantum_state(state)
        return state.get_vector()

    def energy(self, params: np.ndarray, H: np.ndarray) -> float:
        sv = self.statevector(params)
        return float(np.real(sv.conj() @ H @ sv))

    def _build_qulacs(self, params: np.ndarray) -> QCircuit:
        qc = QCircuit(self.n_qubits)
        p = 0
        for gate_type, qubits in self.gates:
            if gate_type == "cnot":
                qc.add_CNOT_gate(int(qubits[0]), int(qubits[1]))
            else:
                angle = float(params[p])
                p += 1
                if gate_type == "rx":
                    qc.add_RotX_gate(qubits[0], angle)
                elif gate_type == "ry":
                    qc.add_RotY_gate(qubits[0], angle)
                elif gate_type == "rz":
                    qc.add_RotZ_gate(qubits[0], angle)
                elif gate_type == "xx":
                    qc.add_multi_Pauli_rotation_gate(list(qubits), [1, 1], angle)
                elif gate_type == "yy":
                    qc.add_multi_Pauli_rotation_gate(list(qubits), [2, 2], angle)
                elif gate_type == "zz":
                    qc.add_multi_Pauli_rotation_gate(list(qubits), [3, 3], angle)
                else:
                    raise ValueError(f"Unsupported TF-QAS native gate '{gate_type}'.")
        return qc

    def compile_to_primitives(self, params: np.ndarray) -> list[dict]:
        """
        Lower the native gate list to primitive RX/RY/RZ/CNOT operations.

        The compiled primitive angles serve as benchmark warm-start parameters.
        They may be further optimised by the shared benchmark optimizer.
        """
        params_arr = np.asarray(params, dtype=np.float64).reshape(-1)
        if params_arr.shape[0] != self.n_params:
            raise ValueError("Parameter vector length does not match native gate count.")

        compiled: list[dict] = []
        p = 0  # index into params_arr (only PARAM_GATES consume a parameter)
        for native_index, (gate_type, qubits) in enumerate(self.gates):
            if gate_type == "cnot":
                # Native CNOT: already primitive, no parameter consumed.
                compiled.append(
                    {
                        "type": "cnot",
                        "ctrl": int(qubits[0]),
                        "targ": int(qubits[1]),
                        "native_gate": "cnot",
                        "native_index": native_index,
                    }
                )
                continue

            angle = float(params_arr[p])
            p += 1

            if gate_type in SINGLE_GATES:
                compiled.append(
                    {
                        "type": "rot",
                        "q": int(qubits[0]),
                        "axis": _NATIVE_AXIS[gate_type],
                        "angle": float(angle),
                        "native_gate": gate_type,
                        "native_index": native_index,
                    }
                )
                continue

            q0, q1 = (int(qubits[0]), int(qubits[1]))
            if gate_type == "zz":
                compiled.extend(
                    [
                        {
                            "type": "cnot",
                            "ctrl": q0,
                            "targ": q1,
                            "native_gate": gate_type,
                            "native_index": native_index,
                        },
                        {
                            "type": "rot",
                            "q": q1,
                            "axis": 3,
                            "angle": float(angle),
                            "native_gate": gate_type,
                            "native_index": native_index,
                        },
                        {
                            "type": "cnot",
                            "ctrl": q0,
                            "targ": q1,
                            "native_gate": gate_type,
                            "native_index": native_index,
                        },
                    ]
                )
            elif gate_type == "xx":
                compiled.extend(
                    [
                        {"type": "rot", "q": q0, "axis": 2, "angle": np.pi / 2, "native_gate": gate_type, "native_index": native_index},
                        {"type": "rot", "q": q1, "axis": 2, "angle": np.pi / 2, "native_gate": gate_type, "native_index": native_index},
                        {"type": "cnot", "ctrl": q0, "targ": q1, "native_gate": gate_type, "native_index": native_index},
                        {"type": "rot", "q": q1, "axis": 3, "angle": float(angle), "native_gate": gate_type, "native_index": native_index},
                        {"type": "cnot", "ctrl": q0, "targ": q1, "native_gate": gate_type, "native_index": native_index},
                        {"type": "rot", "q": q0, "axis": 2, "angle": -np.pi / 2, "native_gate": gate_type, "native_index": native_index},
                        {"type": "rot", "q": q1, "axis": 2, "angle": -np.pi / 2, "native_gate": gate_type, "native_index": native_index},
                    ]
                )
            elif gate_type == "yy":
                compiled.extend(
                    [
                        {"type": "rot", "q": q0, "axis": 1, "angle": np.pi / 2, "native_gate": gate_type, "native_index": native_index},
                        {"type": "rot", "q": q1, "axis": 1, "angle": np.pi / 2, "native_gate": gate_type, "native_index": native_index},
                        {"type": "cnot", "ctrl": q0, "targ": q1, "native_gate": gate_type, "native_index": native_index},
                        {"type": "rot", "q": q1, "axis": 3, "angle": float(angle), "native_gate": gate_type, "native_index": native_index},
                        {"type": "cnot", "ctrl": q0, "targ": q1, "native_gate": gate_type, "native_index": native_index},
                        {"type": "rot", "q": q0, "axis": 1, "angle": -np.pi / 2, "native_gate": gate_type, "native_index": native_index},
                        {"type": "rot", "q": q1, "axis": 1, "angle": -np.pi / 2, "native_gate": gate_type, "native_index": native_index},
                    ]
                )
            else:
                raise ValueError(f"Unsupported TF-QAS native gate '{gate_type}'.")

        return compiled

    def __repr__(self):
        return (
            f"Circuit(n_qubits={self.n_qubits}, n_gates={len(self.gates)}, "
            f"n_params={self.n_params}, n_two_qubit={self.n_two_qubit}, "
            f"path_count={self.path_count})"
        )
