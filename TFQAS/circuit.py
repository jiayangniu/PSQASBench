"""
circuit.py — Circuit template with DAG path counting and qulacs execution.

Gate set: {'rx', 'ry', 'rz', 'cnot'}
  - rx/ry/rz: single-qubit parametric rotations
  - cnot: non-parametric two-qubit gate

DAG path count: number of distinct directed paths from source to sink,
used as the zero-cost structural proxy in TF-QAS Stage 1.
"""

import numpy as np
from collections import defaultdict
from qulacs import QuantumCircuit as QCircuit, QuantumState

PARAM_GATES = frozenset(['rx', 'ry', 'rz'])
TWO_QUBIT_GATES = frozenset(['cnot'])
ALL_GATES = PARAM_GATES | TWO_QUBIT_GATES


class Circuit:
    """
    Quantum circuit template defined as an ordered list of (gate_type, qubits) tuples.
    Parameters are provided externally (not stored in the template).
    """

    def __init__(self, n_qubits: int, gates: list):
        """
        Args:
            n_qubits: number of qubits
            gates: list of (gate_type, qubit_args) tuples
                   gate_type ∈ {'rx', 'ry', 'rz', 'cnot'}
                   qubit_args: tuple of qubit indices, e.g. (0,) or (0, 1)
        """
        self.n_qubits = n_qubits
        self.gates = list(gates)
        self.n_params = sum(1 for g, _ in gates if g in PARAM_GATES)
        self.n_cnot = sum(1 for g, _ in gates if g == 'cnot')
        self._path_count = None

    # ─────────────────────────── DAG path counting ───────────────────────────

    @property
    def path_count(self) -> int:
        """Number of distinct directed paths from source to sink in the DAG."""
        if self._path_count is None:
            self._path_count = self._count_dag_paths()
        return self._path_count

    def _count_dag_paths(self) -> int:
        """
        Build the circuit DAG and count paths via dynamic programming.

        Construction rules:
          - Source node: index -1 (virtual)
          - Gate nodes:  indices 0 … len(gates)-1
          - Sink node:   index len(gates) (virtual)

        For gate i touching qubits q0, q1, …:
          Add edge: last_gate_on[q] → i  for each q
          Update:   last_gate_on[q] = i

        After all gates, connect last_gate_on[q] → sink for each q.

        path_count[sink] = number of paths from source to sink.
        """
        n = len(self.gates)
        if n == 0:
            return 1

        last = [-1] * self.n_qubits          # current tail node per qubit wire
        adj: dict = defaultdict(list)

        for i, (_, qubits) in enumerate(self.gates):
            for q in qubits:
                adj[last[q]].append(i)
            for q in qubits:
                last[q] = i

        sink = n
        for q in range(self.n_qubits):
            adj[last[q]].append(sink)

        # DP in topological order (gates are added in order, so 0..n-1 is valid)
        dp = defaultdict(int)
        dp[-1] = 1
        for i in range(-1, n):
            for j in adj[i]:
                dp[j] += dp[i]

        return dp[sink]

    # ──────────────────────────── qulacs execution ───────────────────────────

    def statevector(self, params: np.ndarray) -> np.ndarray:
        """
        Run the circuit with the given parameters and return the complex statevector.

        Args:
            params: 1-D array of length self.n_params

        Returns:
            Complex numpy array of length 2**n_qubits
        """
        state = QuantumState(self.n_qubits)
        state.set_zero_state()
        qc = self._build_qulacs(params)
        qc.update_quantum_state(state)
        return state.get_vector()

    def energy(self, params: np.ndarray, H: np.ndarray) -> float:
        """
        Compute VQE energy: <ψ(params)|H|ψ(params)>.

        Args:
            params: rotation angles, length self.n_params
            H: Hamiltonian matrix, shape (2**n_qubits, 2**n_qubits)

        Returns:
            Real-valued energy (electronic part only, without energy_shift)
        """
        sv = self.statevector(params)
        return float(np.real(sv.conj() @ H @ sv))

    def _build_qulacs(self, params: np.ndarray) -> QCircuit:
        qc = QCircuit(self.n_qubits)
        p = 0
        for gate_type, qubits in self.gates:
            if gate_type == 'rx':
                qc.add_RX_gate(qubits[0], params[p]); p += 1
            elif gate_type == 'ry':
                qc.add_RY_gate(qubits[0], params[p]); p += 1
            elif gate_type == 'rz':
                qc.add_RZ_gate(qubits[0], params[p]); p += 1
            elif gate_type == 'cnot':
                qc.add_CNOT_gate(qubits[0], qubits[1])
        return qc

    # ─────────────────────────────── utils ───────────────────────────────────

    def __repr__(self):
        return (
            f"Circuit(n_qubits={self.n_qubits}, n_gates={len(self.gates)}, "
            f"n_params={self.n_params}, n_cnot={self.n_cnot}, "
            f"path_count={self.path_count})"
        )
