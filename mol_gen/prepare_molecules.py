import pennylane as qml
from pennylane import qchem
import numpy as np
from pathlib import Path

MOL_DATA_DIR = Path(__file__).resolve().parent.parent / "mol_data"


def generate_mol_data_for_rl(mol_name, symbols, coordinates_angstrom,
                             active_electrons, active_orbitals,
                             geometry_str, mapping='jordan_wigner',
                             charge=0, mult=1):
    """Generate .npz mol data and return the ground-state total energy for config."""
    coordinates_bohr = coordinates_angstrom * 1.8897259886

    H, qubits = qchem.molecular_hamiltonian(
        symbols, coordinates_bohr, charge=charge, mult=mult,
        active_electrons=active_electrons, active_orbitals=active_orbitals,
        mapping=mapping
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

    # build full matrix and diagonalise for exact eigenvalues
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
             energy_shift=energy_shift)

    print(f"[OK] {mol_name}  saved -> {filename}")
    print(f"     fake_min_energy = {total_ground_energy:.8f}\n")

    return total_ground_energy


# =================================================================
# 6-Tier Molecular Diagnostic Suite  (tier definitions in CLAUDE.md)
# =================================================================

# --- L1: Minimalism ---
geom_h2_equil = "H .0 .0 0.0; H .0 .0 0.735"
generate_mol_data_for_rl(
    "L1_H2_Equil", ["H", "H"], np.array([[0,0,0],[0,0,0.735]]),
    2, 2, geometry_str=geom_h2_equil
)

geom_bh = "B .0 .0 0.0; H .0 .0 1.232"
generate_mol_data_for_rl(
    "L1_BH", ["B", "H"], np.array([[0,0,0],[0,0,1.232]]),
    2, 3, geometry_str=geom_bh
)

# --- L2: Asymmetry ---
geom_beh_plus = "Be .0 .0 0.0; H .0 .0 1.312"
generate_mol_data_for_rl(
    "L2_BeH_Plus", ["Be", "H"], np.array([[0,0,0],[0,0,1.312]]),
    2, 2, geometry_str=geom_beh_plus, charge=1
)

geom_lih_equil = "Li .0 .0 0.0; H .0 .0 1.595"
generate_mol_data_for_rl(
    "L2_LiH_Equil", ["Li", "H"], np.array([[0,0,0],[0,0,1.595]]),
    2, 3, geometry_str=geom_lih_equil
)

geom_bf = "B .0 .0 0.0; F .0 .0 1.267"
generate_mol_data_for_rl(
    "L2_BF", ["B", "F"], np.array([[0,0,0],[0,0,1.267]]),
    6, 4, geometry_str=geom_bf
)

# --- L3: Stability ---
geom_heh_plus = "He .0 .0 0.0; H .0 .0 0.774"
generate_mol_data_for_rl(
    "L3_HeH_Plus", ["He", "H"], np.array([[0,0,0],[0,0,0.774]]),
    2, 2, geometry_str=geom_heh_plus, charge=1
)

geom_ch2 = "C .0 .0 0.0; H .0 0.86 0.73; H .0 -0.86 0.73"
generate_mol_data_for_rl(
    "L3_CH2_Singlet", ["C", "H", "H"], np.array([[0,0,0],[0,0.86,0.73],[0,-0.86,0.73]]),
    2, 3, geometry_str=geom_ch2
)

geom_lih_stretch = "Li .0 .0 0.0; H .0 .0 3.500"
generate_mol_data_for_rl(
    "L3_LiH_Stretch", ["Li", "H"], np.array([[0,0,0],[0,0,3.500]]),
    2, 3, geometry_str=geom_lih_stretch
)

geom_h3_triangle = "H .0 .0 0.0; H 1.0 .0 0.0; H 0.5 0.866 0.0"
generate_mol_data_for_rl(
    "L3_H3_Triangle", ["H", "H", "H"], np.array([[0,0,0],[1,0,0],[0.5,0.866,0]]),
    2, 3, geometry_str=geom_h3_triangle, charge=1
)

# --- L4: Representation ---
geom_h2_stretch = "H .0 .0 0.0; H .0 .0 2.5"
generate_mol_data_for_rl(
    "L4_H2_Stretch", ["H", "H"], np.array([[0,0,0],[0,0,2.5]]),
    2, 2, geometry_str=geom_h2_stretch
)

geom_h3_linear = "H .0 .0 0.0; H .0 .0 1.0; H .0 .0 2.0"
generate_mol_data_for_rl(
    "L4_H3_Linear", ["H", "H", "H"], np.array([[0,0,0],[0,0,1],[0,0,2]]),
    2, 3, geometry_str=geom_h3_linear, charge=1
)

geom_h2o = "O .0 .0 0.0; H .0 0.757 0.586; H .0 -0.757 0.586"
generate_mol_data_for_rl(
    "L4_H2O_StrongCorr", ["O", "H", "H"], np.array([[0,0,0],[0,0.757,0.586],[0,-0.757,0.586]]),
    4, 4, geometry_str=geom_h2o
)

# --- L5: Topology (same geometry as L4 H3/H4, but restricted to 1D nearest-neighbor at runtime) ---
generate_mol_data_for_rl(
    "L5_H3_Linear", ["H", "H", "H"], np.array([[0,0,0],[0,0,1],[0,0,2]]),
    2, 3, geometry_str=geom_h3_linear, charge=1
)

geom_h4_chain = "H .0 .0 0.0; H .0 .0 1.0; H .0 .0 2.0; H .0 .0 3.0"
generate_mol_data_for_rl(
    "L5_H4_Chain", ["H", "H", "H", "H"], np.array([[0,0,0],[0,0,1],[0,0,2],[0,0,3]]),
    4, 4, geometry_str=geom_h4_chain
)

# --- L6: Scalability ---
geom_beh2 = "Be .0 .0 0.0; H .0 .0 1.326; H .0 .0 -1.326"
generate_mol_data_for_rl(
    "L6_BeH2_Scalability", ["Be", "H", "H"], np.array([[0,0,0],[0,0,1.326],[0,0,-1.326]]),
    2, 5, geometry_str=geom_beh2
)
