import pennylane as qml
from pennylane import qchem
import numpy as np
from pathlib import Path


OUT_DIR = Path(__file__).resolve().parent.parent / "mol_data_h2_basis"
OUT_DIR.mkdir(exist_ok=True)


def generate_h2_case(
    mol_name,
    bond_length_angstrom,
    basis,
    active_electrons,
    active_orbitals,
    mapping="jordan_wigner",
    save_matrix=True,
):
    """Generate one H2 Hamiltonian case and save a small .npz summary.

    This is a lightweight exploration script for Level-6 candidate generation:
    we keep the molecule fixed (H2) and vary the basis/active-space settings to
    see whether the qubit count can grow in a controlled way.
    """
    symbols = ["H", "H"]
    coordinates_angstrom = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, bond_length_angstrom]])
    coordinates_bohr = coordinates_angstrom * 1.8897259886

    H, n_qubits = qchem.molecular_hamiltonian(
        symbols,
        coordinates_bohr,
        charge=0,
        mult=1,
        active_electrons=active_electrons,
        active_orbitals=active_orbitals,
        mapping=mapping,
        basis=basis,
    )

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
        "bond_length_angstrom": bond_length_angstrom,
        "active_electrons": active_electrons,
        "active_orbitals": active_orbitals,
        "n_qubits": n_qubits,
        "weights": np.array(electronic_coeffs),
        "pauli_strings": np.array([str(op) for op in electronic_ops]),
        "energy_shift": energy_shift,
    }

    if save_matrix:
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
    np.savez(filename, **save_payload)

    print(
        f"[OK] {mol_name:20s} basis={basis:10s} ao={active_orbitals:<2d} "
        f"-> {n_qubits:2d} qubits"
        + (
            f", E0={total_ground_energy:.8f}"
            if save_matrix
            else ""
        )
    )

    return {
        "mol_name": mol_name,
        "basis": basis,
        "active_orbitals": active_orbitals,
        "n_qubits": n_qubits,
        "filename": str(filename),
        "ground_energy": total_ground_energy,
    }


def main():
    bond_length_angstrom = 0.735
    active_electrons = 2

    # Start with common basis families. We keep active_electrons fixed and vary
    # the active orbitals to probe whether H2 can provide a clean qubit ladder.
    cases = [
        ("H2_STO3G", "sto-3g", 2),
        ("H2_631G", "6-31g", 3),
        ("H2_6311G", "6-311g", 4),
        ("H2_CCPVDZ", "cc-pvdz", 5),
        ("H2_CCPVTZ", "cc-pvtz", 6),
    ]

    print("=== H2 basis sweep for Level-6 exploration ===")
    print(f"bond length = {bond_length_angstrom} Angstrom")
    print(f"active electrons = {active_electrons}")
    print()

    results = []
    for mol_name, basis, active_orbitals in cases:
        try:
            result = generate_h2_case(
                mol_name=mol_name,
                bond_length_angstrom=bond_length_angstrom,
                basis=basis,
                active_electrons=active_electrons,
                active_orbitals=active_orbitals,
            )
            results.append(result)
        except Exception as exc:
            print(
                f"[FAIL] {mol_name:20s} basis={basis:10s} ao={active_orbitals:<2d} "
                f"-> {exc}"
            )

    print()
    print("=== Summary ===")
    for result in results:
        print(
            f"{result['mol_name']:20s} basis={result['basis']:10s} "
            f"ao={result['active_orbitals']:<2d} -> {result['n_qubits']:2d} qubits"
        )


if __name__ == "__main__":
    main()
