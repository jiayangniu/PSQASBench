import threading
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pennylane as qml
from pennylane import qchem

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - optional dependency
    tqdm = None


OUT_DIR = Path(__file__).resolve().parent.parent / "mol_data_beh2_basis"
OUT_DIR.mkdir(exist_ok=True)

TARGET_QUBITS = 14
TARGET_ACTIVE_ELECTRONS = 2
TARGET_ACTIVE_ORBITALS = TARGET_QUBITS // 2

# Ordered from the most chemistry-standard first to broader fallbacks.
# The script stops after the first successful 14-qubit case.
BASIS_CANDIDATES = [
    "cc-pvtz",
    "aug-cc-pvdz",
    "def2-tzvp",
]


def _iter_with_progress(items, desc):
    if tqdm is None:
        return items
    return tqdm(items, desc=desc, unit="basis")


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


def generate_beh2_case(
    mol_name,
    basis,
    active_electrons,
    active_orbitals,
    mapping="jordan_wigner",
    save_matrix=False,
):
    """Generate one BeH2 Hamiltonian case, with progress for long stages."""
    symbols = ["Be", "H", "H"]
    coordinates_angstrom = np.array(
        [[0.0, 0.0, 0.0], [0.0, 0.0, 1.326], [0.0, 0.0, -1.326]]
    )
    coordinates_bohr = coordinates_angstrom * 1.8897259886

    case_start = time.perf_counter()
    print(
        f"[TRY] {mol_name:20s} basis={basis:12s} "
        f"ao={active_orbitals:<2d} target={2 * active_orbitals:2d}q",
        flush=True,
    )

    with timed_stage("build molecular Hamiltonian", heartbeat=True, interval_s=20):
        H, n_qubits = qchem.molecular_hamiltonian(
            symbols,
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
            if op.name in ["Identity", "I"]:
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

    if save_matrix:
        with timed_stage("materialize Hamiltonian matrix + diagonalize", heartbeat=True, interval_s=20):
            dim = 2 ** n_qubits
            ham_mat = np.zeros((dim, dim), dtype=np.complex128)
            for c, o in zip(electronic_coeffs, electronic_ops):
                ham_mat += c * qml.matrix(o, wire_order=range(n_qubits))
            eigvals = np.linalg.eigh(ham_mat)[0]
            save_payload["hamiltonian"] = ham_mat
            save_payload["eigvals"] = eigvals
            total_ground_energy = float(eigvals[0].real + energy_shift)
    else:
        total_ground_energy = np.nan

    filename = OUT_DIR / (
        f"{mol_name}_{n_qubits}q_{basis}_ao{active_orbitals}_{mapping}.npz"
    )
    with timed_stage("save npz payload", heartbeat=False):
        np.savez(filename, **save_payload)

    elapsed = time.perf_counter() - case_start
    print(
        f"[OK]  {mol_name:20s} basis={basis:12s} ao={active_orbitals:<2d} "
        f"-> {n_qubits:2d} qubits in {elapsed:.1f}s"
        + (f", E0={total_ground_energy:.8f}" if save_matrix else ""),
        flush=True,
    )

    return {
        "mol_name": mol_name,
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
    save_matrix = False

    print("=== BeH2 targeted 14-qubit search for Level-6 exploration ===")
    print(f"target qubits      = {TARGET_QUBITS}")
    print(f"active electrons   = {active_electrons}")
    print(f"active orbitals    = {active_orbitals}")
    print(f"save full matrix   = {save_matrix}")
    print("candidate bases    = " + ", ".join(BASIS_CANDIDATES))
    print()

    results = []
    failures = []

    for basis in _iter_with_progress(BASIS_CANDIDATES, desc="Trying basis sets"):
        mol_name = f"BeH2_{TARGET_QUBITS}q"
        try:
            result = generate_beh2_case(
                mol_name=mol_name,
                basis=basis,
                active_electrons=active_electrons,
                active_orbitals=active_orbitals,
                save_matrix=save_matrix,
            )
            results.append(result)
            print()
            print(f"[STOP] First successful {TARGET_QUBITS}-qubit BeH2 found with basis={basis}")
            break
        except Exception as exc:
            failures.append((basis, str(exc)))
            print(
                f"[FAIL] {mol_name:20s} basis={basis:12s} ao={active_orbitals:<2d} "
                f"-> {exc}",
                flush=True,
            )
            print()

    print("=== Summary ===")
    if results:
        result = results[0]
        print(
            f"SUCCESS  {result['mol_name']:20s} basis={result['basis']:12s} "
            f"ao={result['active_orbitals']:<2d} -> {result['n_qubits']:2d} qubits "
            f"in {result['elapsed_s']:.1f}s"
        )
    else:
        print(f"No {TARGET_QUBITS}-qubit BeH2 case was generated.")

    if failures:
        print()
        print("Failure log:")
        for basis, err in failures:
            print(f"  basis={basis:12s} -> {err}")


if __name__ == "__main__":
    main()
