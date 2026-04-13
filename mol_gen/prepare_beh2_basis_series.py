"""Generate BeH2 basis-set benchmark cases for the PSQASBench scalability track.

This script is currently configured to search for the first basis that reaches
the target qubit count, save the resulting `.npz` file, and stop.  Small cases
store the full dense Hamiltonian matrix; larger cases switch to sparse
diagonalization and save only the low-lying eigenvalues together with the Pauli
decomposition.
"""

import threading
import time
import tempfile
import os
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
import pennylane as qml
from pennylane import qchem

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - optional dependency
    tqdm = None

# Save alongside the main benchmark molecule files so the same filenames can be
# consumed directly by the benchmark registry and configuration files.
OUT_DIR = Path(__file__).resolve().parent.parent / "mol_data"
OUT_DIR.mkdir(exist_ok=True)

TARGET_QUBITS = 14
# Optional high-qubit extension: CAS(4e, 7o), obtained by freezing the Be 1s
# core and keeping the four valence electrons active over seven orbitals.
TARGET_ACTIVE_ELECTRONS = 4
TARGET_ACTIVE_ORBITALS = TARGET_QUBITS // 2

# Try heavier basis sets in order and stop after the first successful 14-qubit
# construction.  This keeps the script practical for one-off data generation.
BASIS_CANDIDATES = [
    "cc-pvdz",
    "cc-pvtz",
    "def2-tzvp",
]

# Linear BeH2 at equilibrium geometry (Angstrom), shared across all basis sets.
_SYMBOLS = ["Be", "H", "H"]
_COORDS_ANGSTROM = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.326], [0.0, 0.0, -1.326]])
_GEOM_TAG = "Be_.0_.0_0.0;_H_.0_.0_1.326;_H_.0_.0_-1.326"

# Above this threshold the dense matrix becomes too large to store comfortably,
# so we diagonalize a sparse Hamiltonian instead.
_DENSE_DIAG_MAX_QUBITS = 12
# Number of low-lying eigenvalues retained for large sparse runs.
_SPARSE_EIGSH_K = 20
# For >12q we still materialize a dense Hamiltonian for training compatibility,
# but build it through a disk-backed memmap to avoid a huge RAM spike.
_SAVE_TRAINING_HAMILTONIAN_FOR_LARGE_CASES = True
_LARGE_CASE_HAMILTONIAN_DTYPE = np.complex64


def _basis_to_tag(basis: str) -> str:
    """Map basis name to the uppercase tag used in benchmark filenames."""
    return basis.upper().replace("-", "").replace("_", "")


def _iter_with_progress(items, desc):
    if tqdm is None:
        return items
    return tqdm(items, desc=desc, unit="basis")


# ---------------------------------------------------------------------------
# Sparse Pauli matrix helpers
# ---------------------------------------------------------------------------

_P1 = {
    "PauliX": sp.csr_matrix(np.array([[0, 1], [1, 0]], dtype=np.complex128)),
    "PauliY": sp.csr_matrix(np.array([[0, -1j], [1j, 0]], dtype=np.complex128)),
    "PauliZ": sp.csr_matrix(np.array([[1, 0], [0, -1]], dtype=np.complex128)),
    "X": sp.csr_matrix(np.array([[0, 1], [1, 0]], dtype=np.complex128)),
    "Y": sp.csr_matrix(np.array([[0, -1j], [1j, 0]], dtype=np.complex128)),
    "Z": sp.csr_matrix(np.array([[1, 0], [0, -1]], dtype=np.complex128)),
}
_I2 = sp.eye(2, format="csr", dtype=np.complex128)


def _op_to_sparse(op, n_qubits: int):
    """Return a scipy CSR sparse matrix for a PennyLane Pauli op."""
    wire_mats = [None] * n_qubits  # None → identity on that qubit

    def _assign(sub):
        w = int(sub.wires[0])
        wire_mats[w] = _P1.get(sub.name)

    name = op.name
    if name in _P1:
        _assign(op)
    elif hasattr(op, "obs"):  # Tensor (older PennyLane API)
        for sub in op.obs:
            _assign(sub)
    elif hasattr(op, "operands"):  # Prod (newer PennyLane API)
        for sub in op.operands:
            if sub.name in _P1:
                _assign(sub)

    result = wire_mats[0] if wire_mats[0] is not None else _I2
    for m in wire_mats[1:]:
        result = sp.kron(result, (m if m is not None else _I2), format="csr")
    return result


def _build_sparse_hamiltonian(coeffs, ops, n_qubits: int):
    """Sum Pauli terms into a single CSR sparse Hamiltonian matrix."""
    dim = 2 ** n_qubits
    H_sparse = sp.csr_matrix((dim, dim), dtype=np.complex128)
    for c, op in zip(coeffs, ops):
        H_sparse = H_sparse + float(c) * _op_to_sparse(op, n_qubits)
    return H_sparse


def _build_dense_hamiltonian_memmap(coeffs, ops, n_qubits: int, dtype):
    """Build a dense Hamiltonian in a temporary memmap and return memmap + path."""
    dim = 2 ** n_qubits
    fd, tmp_path = tempfile.mkstemp(
        prefix=f"beh2_{n_qubits}q_ham_",
        suffix=".dat",
        dir=str(OUT_DIR),
    )
    os.close(fd)

    ham_map = np.memmap(tmp_path, mode="w+", dtype=dtype, shape=(dim, dim))
    ham_map[:] = 0
    for coeff, op in zip(coeffs, ops):
        ham_map += np.asarray(
            coeff * qml.matrix(op, wire_order=range(n_qubits)),
            dtype=dtype,
        )
    ham_map.flush()
    return ham_map, Path(tmp_path)


# ---------------------------------------------------------------------------
# Progress utilities
# ---------------------------------------------------------------------------


class StageTicker:
    """Periodic liveness messages for long chemistry stages."""

    def __init__(self, label, interval_s=30):
        self.label = label
        self.interval_s = interval_s
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._start = None

    def _run(self):
        while not self._stop.wait(self.interval_s):
            elapsed = time.perf_counter() - self._start
            print(f"    ... {self.label} still running ({elapsed:.0f}s elapsed)", flush=True)

    def start(self):
        self._start = time.perf_counter()
        self._thread.start()

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=0.1)


@contextmanager
def timed_stage(label, heartbeat=True, interval_s=30):
    print(f"  -> {label} ...", flush=True)
    start = time.perf_counter()
    ticker = StageTicker(label, interval_s=interval_s) if heartbeat else None
    if ticker is not None:
        ticker.start()
    try:
        yield
    finally:
        if ticker is not None:
            ticker.stop()
        elapsed = time.perf_counter() - start
        print(f"  <- {label} done in {elapsed:.1f}s", flush=True)


# ---------------------------------------------------------------------------
# Main generation function
# ---------------------------------------------------------------------------


def generate_beh2_case(
    basis,
    active_electrons,
    active_orbitals,
    mapping="jordan_wigner",
    save_eigvals=True,
):
    """Generate one BeH2 Hamiltonian case and save it to `mol_data/`."""
    coordinates_bohr = _COORDS_ANGSTROM * 1.8897259886

    case_start = time.perf_counter()
    print(
        f"[TRY] BeH2  basis={basis:12s} "
        f"CAS({active_electrons}e,{active_orbitals}o)  target={2 * active_orbitals:2d}q",
        flush=True,
    )

    with timed_stage("build molecular Hamiltonian", heartbeat=True, interval_s=20):
        H, n_qubits = qchem.molecular_hamiltonian(
            _SYMBOLS,
            coordinates_bohr,
            charge=0,
            mult=1,
            active_electrons=active_electrons,
            active_orbitals=active_orbitals,
            mapping=mapping,
            basis=basis,
            load_data=True,
        )

    if n_qubits != 2 * active_orbitals:
        raise RuntimeError(
            f"unexpected qubit count: got {n_qubits}, expected {2 * active_orbitals}"
        )

    with timed_stage("split identity / electronic terms", heartbeat=False):
        coeffs, ops = H.terms()
        energy_shift = 0.0
        electronic_coeffs, electronic_ops = [], []
        for coeff, op in zip(coeffs, ops):
            if op.name in ("Identity", "I"):
                energy_shift += float(coeff)
            else:
                electronic_coeffs.append(float(coeff))
                electronic_ops.append(op)

    save_payload = {
        "basis": basis,
        "active_electrons": active_electrons,
        "active_orbitals": active_orbitals,
        "n_qubits": n_qubits,
        "weights": np.array(electronic_coeffs),
        "pauli_strings": np.array([str(op) for op in electronic_ops]),
        "energy_shift": energy_shift,
    }

    total_ground_energy = np.nan

    dense_ham_tmp_path = None

    if save_eigvals:
        if n_qubits <= _DENSE_DIAG_MAX_QUBITS:
            # Dense diagonalization is still practical for the smaller cases.
            with timed_stage("dense Hamiltonian matrix + full diagonalization", heartbeat=True, interval_s=20):
                dim = 2 ** n_qubits
                ham_mat = np.zeros((dim, dim), dtype=np.complex128)
                for c, o in zip(electronic_coeffs, electronic_ops):
                    ham_mat += c * qml.matrix(o, wire_order=range(n_qubits))
                eigvals = np.linalg.eigh(ham_mat)[0]
                save_payload["hamiltonian"] = ham_mat
                save_payload["eigvals"] = eigvals
                total_ground_energy = float(eigvals[0].real + energy_shift)
        else:
            # For 14 qubits, ground-state estimation stays on the sparse path.
            # If requested, we additionally materialize a dense matrix for the
            # current training stack, which expects a "hamiltonian" payload.
            k = min(_SPARSE_EIGSH_K, 2 ** n_qubits - 2)
            with timed_stage(
                f"sparse Hamiltonian + eigsh (k={k} lowest eigenvalues)",
                heartbeat=True,
                interval_s=30,
            ):
                sparse_H = _build_sparse_hamiltonian(electronic_coeffs, electronic_ops, n_qubits)
                eigvals_elec, _ = eigsh(sparse_H, k=k, which="SA")
                eigvals_elec = np.sort(eigvals_elec.real)
                save_payload["eigvals"] = eigvals_elec
                total_ground_energy = float(eigvals_elec[0] + energy_shift)

            if _SAVE_TRAINING_HAMILTONIAN_FOR_LARGE_CASES:
                with timed_stage(
                    f"dense Hamiltonian memmap for training ({_LARGE_CASE_HAMILTONIAN_DTYPE.__name__})",
                    heartbeat=True,
                    interval_s=30,
                ):
                    ham_mat, dense_ham_tmp_path = _build_dense_hamiltonian_memmap(
                        electronic_coeffs,
                        electronic_ops,
                        n_qubits,
                        dtype=_LARGE_CASE_HAMILTONIAN_DTYPE,
                    )
                    save_payload["hamiltonian"] = ham_mat

    basis_tag = _basis_to_tag(basis)
    mol_name = f"L6_BeH2_{basis_tag}_{n_qubits}q_geom_{_GEOM_TAG}_{mapping}"
    filename = OUT_DIR / f"{mol_name}.npz"

    try:
        with timed_stage("save npz payload", heartbeat=False):
            np.savez(filename, **save_payload)
    finally:
        if dense_ham_tmp_path is not None and dense_ham_tmp_path.exists():
            dense_ham_tmp_path.unlink()

    elapsed = time.perf_counter() - case_start
    print(
        f"[OK]  BeH2  basis={basis:12s}  {n_qubits:2d}q  {len(electronic_coeffs)} Pauli terms  "
        f"{elapsed:.1f}s"
        + (f"  E0={total_ground_energy:.8f} Ha" if save_eigvals else ""),
        flush=True,
    )

    return {
        "basis": basis,
        "active_orbitals": active_orbitals,
        "n_qubits": n_qubits,
        "filename": str(filename),
        "ground_energy": total_ground_energy,
        "elapsed_s": elapsed,
    }


def main():
    active_electrons = TARGET_ACTIVE_ELECTRONS
    active_orbitals = TARGET_ACTIVE_ORBITALS
    save_eigvals = True

    print("=== BeH2 14-qubit Hamiltonian generation (Level-6) ===")
    print(f"target qubits      = {TARGET_QUBITS}")
    print(f"active electrons   = {active_electrons}  (CAS: freeze Be 1s core)")
    print(f"active orbitals    = {active_orbitals}")
    print(f"save eigvals       = {save_eigvals}")
    print(f"sparse threshold   = >{_DENSE_DIAG_MAX_QUBITS}q uses eigsh (k={_SPARSE_EIGSH_K})")
    print("candidate bases    = " + ", ".join(BASIS_CANDIDATES))
    print()

    results = []
    failures = []

    for basis in _iter_with_progress(BASIS_CANDIDATES, desc="Trying basis sets"):
        try:
            result = generate_beh2_case(
                basis=basis,
                active_electrons=active_electrons,
                active_orbitals=active_orbitals,
                save_eigvals=save_eigvals,
            )
            results.append(result)
            print()
            print(f"[STOP] First successful {TARGET_QUBITS}-qubit BeH2 found with basis={basis}")
            break
        except Exception as exc:
            failures.append((basis, str(exc)))
            print(
                f"[FAIL] BeH2  basis={basis:12s}  ao={active_orbitals:<2d}  -> {exc}",
                flush=True,
            )
            print()

    print("=== Summary ===")
    if results:
        r = results[0]
        print(
            f"SUCCESS  basis={r['basis']:12s}  ao={r['active_orbitals']:<2d}  "
            f"-> {r['n_qubits']:2d}q  in {r['elapsed_s']:.1f}s"
        )
        print(f"         saved: {r['filename']}")
    else:
        print(f"No {TARGET_QUBITS}-qubit BeH2 case was generated.")

    if failures:
        print()
        print("Failure log:")
        for basis, err in failures:
            print(f"  basis={basis:12s} -> {err}")


if __name__ == "__main__":
    main()
