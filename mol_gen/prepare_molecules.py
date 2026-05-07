"""Generate benchmark molecular Hamiltonians in the full qubit Hilbert space.

The saved `.npz` files contain:
  - the electronic Hamiltonian matrix (the scalar identity shift is stored separately),
  - the Pauli-decomposition coefficients and operator strings,
  - the exact eigenvalues of the full, unrestricted qubit-space Hamiltonian,
  - metadata fields: active_electrons, active_orbitals, basis, n_qubits.

These files are convenient for QAS / VQE experiments that work directly in the
full computational basis.  For spectra restricted to a fixed particle-number
and spin sector, see `prepare_molecules_physical.py`.

Running this script as __main__ regenerates all T1-T4 benchmark molecules
(7 molecules, 4-8 qubits).  For the T5 BeH2 basis-set scalability series
(8-14 qubits) use `prepare_beh2_basis_series.py` instead, which handles the
sparse diagonalization required for the 14-qubit case.

Final benchmark suite
---------------------
Tier  Molecule          q   active_electrons  active_orbitals  basis
T1    BeH2_STO3G        6   2                 3                sto-3g
T1    LiH_Equil         6   2                 3                sto-3g
T2    CH2_Singlet       8   2                 4                sto-3g
T3    H2_Stretch        4   2                 2                sto-3g
T3    H2O_StrongCorr    8   4                 4                sto-3g
T3    H4_Chain          8   4                 4                sto-3g
T4    H3_Linear         6   2                 3  charge=+1     sto-3g
T5    BeH2_631G         8   2                 4                6-31g    (*see prepare_beh2_basis_series.py)
T5    BeH2_6311G        10  2                 5                6-311g   (*see prepare_beh2_basis_series.py)
T5    BeH2_CCPVDZ       12  2                 6                cc-pvdz  (*see prepare_beh2_basis_series.py)
T5    BeH2_CCPVDZ       14  4                 7                cc-pvdz  (*see prepare_beh2_basis_series.py)
"""

import pennylane as qml
from pennylane import qchem
import numpy as np
from pathlib import Path

MOL_DATA_DIR = Path(__file__).resolve().parent.parent / "mol_data"
MOL_DATA_DIR.mkdir(exist_ok=True)


def generate_mol_data_for_rl(mol_name, symbols, coordinates_angstrom,
                             active_electrons, active_orbitals,
                             geometry_str, mapping='jordan_wigner',
                             charge=0, mult=1, basis='sto-3g'):
    """Build and save one benchmark molecule in the full qubit space.

    Parameters
    ----------
    mol_name : str
        Molecule label used in the output filename.
    symbols : list[str]
        Atomic symbols in the same order as `coordinates_angstrom`.
    coordinates_angstrom : np.ndarray
        Cartesian geometry in Angstrom with shape `(n_atoms, 3)`.
    active_electrons : int
        Number of electrons retained in the active space.
    active_orbitals : int
        Number of spatial orbitals retained in the active space.
    geometry_str : str
        Human-readable geometry tag copied into the filename for traceability.
    mapping : str, optional
        Fermion-to-qubit mapping passed to PennyLane.
    charge : int, optional
        Total molecular charge.
    mult : int, optional
        Spin multiplicity ``2S + 1``.
    basis : str, optional
        Gaussian basis set name (default ``'sto-3g'``).

    Returns
    -------
    float
        Ground-state total energy, i.e. the minimum electronic eigenvalue plus
        the stored scalar identity shift.
    """
    coordinates_bohr = coordinates_angstrom * 1.8897259886

    H, qubits = qchem.molecular_hamiltonian(
        symbols, coordinates_bohr, charge=charge, mult=mult,
        active_electrons=active_electrons, active_orbitals=active_orbitals,
        mapping=mapping, basis=basis
    )

    coeffs, ops = H.terms()
    energy_shift = 0.0
    electronic_coeffs, electronic_ops = [], []

    for coeff, op in zip(coeffs, ops):
        if op.name in ['Identity', 'I']:
            energy_shift += float(coeff)
        else:
            electronic_coeffs.append(float(coeff))
            electronic_ops.append(op)

    # Keep the saved matrix purely electronic and store the scalar shift
    # separately so downstream code can decide when to add it back.
    dim = 2 ** qubits
    ham_mat = np.zeros((dim, dim), dtype=np.complex128)
    for c, o in zip(electronic_coeffs, electronic_ops):
        ham_mat += c * qml.matrix(o, wire_order=range(qubits))

    eigvals, _ = np.linalg.eigh(ham_mat)
    total_ground_energy = min(eigvals).real + energy_shift

    processed_geometry = geometry_str.replace(" ", "_")
    filename = MOL_DATA_DIR / f"{mol_name}_{qubits}q_geom_{processed_geometry}_{mapping}.npz"

    np.savez(filename,
             hamiltonian=ham_mat,
             weights=np.array(electronic_coeffs),
             pauli_strings=np.array([str(op) for op in electronic_ops]),
             eigvals=eigvals,
             energy_shift=energy_shift,
             active_electrons=active_electrons,
             active_orbitals=active_orbitals,
             n_qubits=qubits,
             basis=basis)

    print(f"[OK] {mol_name}  saved -> {filename}")
    print(f"     ground_state_energy = {total_ground_energy:.8f}\n")

    return total_ground_energy


# ==========================================================================
# Geometry definitions — all coordinates in Angstrom
# ==========================================================================

# --- T1 / L1: Minimalism ---
geom_beh2_sto3g = "Be_.0_.0_0.0;_H_.0_.0_1.326;_H_.0_.0_-1.326"

geom_h2_equil = "H_.0_.0_0.0;_H_.0_.0_0.735"
# generate_mol_data_for_rl(
#     "L1_H2_Equil", ["H", "H"], np.array([[0,0,0],[0,0,0.735]]),
#     2, 2, geometry_str=geom_h2_equil
# )

geom_bh = "B_.0_.0_0.0;_H_.0_.0_1.232"
# generate_mol_data_for_rl(
#     "L1_BH", ["B", "H"], np.array([[0,0,0],[0,0,1.232]]),
#     2, 3, geometry_str=geom_bh
# )

# --- T1 / L2: Asymmetry ---
geom_lih_equil = "Li_.0_.0_0.0;_H_.0_.0_1.595"

geom_beh_plus = "Be_.0_.0_0.0;_H_.0_.0_1.312"
# generate_mol_data_for_rl(
#     "L2_BeH_Plus", ["Be", "H"], np.array([[0,0,0],[0,0,1.312]]),
#     2, 2, geometry_str=geom_beh_plus, charge=1
# )

geom_bf = "B_.0_.0_0.0;_F_.0_.0_1.267"
# generate_mol_data_for_rl(
#     "L2_BF", ["B", "F"], np.array([[0,0,0],[0,0,1.267]]),
#     6, 4, geometry_str=geom_bf
# )

# --- T2 / L3: Stability ---
geom_ch2 = "C_.0_.0_0.0;_H_.0_0.86_0.73;_H_.0_-0.86_0.73"

geom_heh_plus = "He_.0_.0_0.0;_H_.0_.0_0.774"
# generate_mol_data_for_rl(
#     "L3_HeH_Plus", ["He", "H"], np.array([[0,0,0],[0,0,0.774]]),
#     2, 2, geometry_str=geom_heh_plus, charge=1
# )

geom_lih_stretch = "Li_.0_.0_0.0;_H_.0_.0_3.500"
# generate_mol_data_for_rl(
#     "L3_LiH_Stretch", ["Li", "H"], np.array([[0,0,0],[0,0,3.500]]),
#     2, 3, geometry_str=geom_lih_stretch
# )

geom_h3_triangle = "H_.0_.0_0.0;_H_1.0_.0_0.0;_H_0.5_0.866_0.0"
# generate_mol_data_for_rl(
#     "L3_H3_Triangle", ["H", "H", "H"], np.array([[0,0,0],[1,0,0],[0.5,0.866,0]]),
#     2, 3, geometry_str=geom_h3_triangle, charge=1
# )

# --- T3 / L4: Representation ---
geom_h2_stretch = "H_.0_.0_0.0;_H_.0_.0_2.5"
geom_h2o = "O_.0_.0_0.0;_H_.0_1.186_0.918;_H_.0_-1.186_0.918"

geom_h3_linear = "H_.0_.0_0.0;_H_.0_.0_1.0;_H_.0_.0_2.0"
# (same geometry used for both L4 and L5 H3_Linear; L5 is the T4 entry)

# --- T3 / L5: Topology ---
geom_h4_chain = "H_.0_.0_0.0;_H_.0_.0_1.0;_H_.0_.0_2.0;_H_.0_.0_3.0"

# --- T4 / L5: Topology (H3 linear chain, cationic) ---
# H3_Linear uses charge=+1 → 2 active electrons over 3 orbitals → 6 qubits

# --- T5 / L6: Scalability (BeH2 basis-set series) ---
# Generated by prepare_beh2_basis_series.py — the 14-qubit case requires
# sparse diagonalization and a dedicated generation script.
#
# generate_beh2_case(basis='6-31g',   active_electrons=2, active_orbitals=4)  # 8q
# generate_beh2_case(basis='6-311g',  active_electrons=2, active_orbitals=5)  # 10q
# generate_beh2_case(basis='cc-pvdz', active_electrons=2, active_orbitals=6)  # 12q
# generate_beh2_case(basis='cc-pvdz', active_electrons=4, active_orbitals=7)  # 14q


# ==========================================================================
# Generate all T1–T4 benchmark molecules
# ==========================================================================

if __name__ == "__main__":
    print("=== PSQASBench T1–T4 molecule generation ===\n")

    # -- T1: BeH2 STO-3G (6q) ------------------------------------------
    # CAS(2e, 3o): freeze Be 1s core (2e); 2 valence electrons active
    generate_mol_data_for_rl(
        "L1_BeH2_STO3G",
        ["Be", "H", "H"],
        np.array([[0, 0, 0], [0, 0, 1.326], [0, 0, -1.326]]),
        active_electrons=2, active_orbitals=3,
        geometry_str=geom_beh2_sto3g,
        basis='sto-3g',
    )

    # -- T1: LiH equilibrium (6q) -----------------------------------------
    # CAS(2e, 3o): freeze Li 1s core (2e); 2 valence electrons active
    generate_mol_data_for_rl(
        "L2_LiH_Equil",
        ["Li", "H"],
        np.array([[0, 0, 0], [0, 0, 1.595]]),
        active_electrons=2, active_orbitals=3,
        geometry_str=geom_lih_equil,
        basis='sto-3g',
    )

    # -- T2: CH2 singlet (8q) ---------------------------------------------
    # CAS(2e, 4o): freeze C 1s + 2 doubly-occupied MOs (6e core); 2 active
    generate_mol_data_for_rl(
        "L3_CH2_Singlet",
        ["C", "H", "H"],
        np.array([[0, 0, 0], [0, 0.86, 0.73], [0, -0.86, 0.73]]),
        active_electrons=2, active_orbitals=4,
        geometry_str=geom_ch2,
        basis='sto-3g',
    )

    # -- T3: H2 stretched (4q) --------------------------------------------
    # CAS(2e, 2o): full valence active space; bond length 2.5 Å (strong corr.)
    generate_mol_data_for_rl(
        "L4_H2_Stretch",
        ["H", "H"],
        np.array([[0, 0, 0], [0, 0, 2.5]]),
        active_electrons=2, active_orbitals=2,
        geometry_str=geom_h2_stretch,
        basis='sto-3g',
    )

    # -- T3: H2O strong-correlation (8q) ----------------------------------
    # CAS(4e, 4o): freeze 6 core electrons; bent geometry near Cs symmetry
    generate_mol_data_for_rl(
        "L4_H2O_StrongCorr",
        ["O", "H", "H"],
        np.array([[0, 0, 0], [0, 1.186, 0.918], [0, -1.186, 0.918]]),
        active_electrons=4, active_orbitals=4,
        geometry_str=geom_h2o,
        basis='sto-3g',
    )

    # -- T3: H4 linear chain (8q) -----------------------------------------
    # CAS(4e, 4o): full valence; 1 Å spacing; topology-induced frustration
    generate_mol_data_for_rl(
        "L5_H4_Chain",
        ["H", "H", "H", "H"],
        np.array([[0, 0, 0], [0, 0, 1], [0, 0, 2], [0, 0, 3]]),
        active_electrons=4, active_orbitals=4,
        geometry_str=geom_h4_chain,
        basis='sto-3g',
    )

    # -- T4: H3 linear chain cationic (6q) --------------------------------
    # CAS(2e, 3o): H3+ (charge=+1); 1 Å spacing; degenerate ground state
    generate_mol_data_for_rl(
        "L5_H3_Linear",
        ["H", "H", "H"],
        np.array([[0, 0, 0], [0, 0, 1], [0, 0, 2]]),
        active_electrons=2, active_orbitals=3,
        geometry_str=geom_h3_linear,
        basis='sto-3g',
        charge=1,
    )

    print("=== T1–T4 generation complete ===")
    print("For T5 BeH2 scalability series (8–14 qubits) run:")
    print("  python mol_gen/prepare_beh2_basis_series.py")
