"""
prepare_molecules_physical.py

Generates molecular Hamiltonian data restricted to the physically correct
particle-number and spin sector.

Problem with the original prepare_molecules.py
-----------------------------------------------
The original code diagonalises the full 2^n x 2^n Hamiltonian matrix, which
includes ALL possible electron counts (0, 1, 2, ..., n_qubits electrons).
For charged molecules this means the stored eigvals[0] can correspond to a
different charge state than intended.  For all molecules the stored eigenvalues
mix different spin sectors (Sz values), introducing artificial degeneracies.

What this script does differently
----------------------------------
After building the full Hamiltonian matrix we project onto the sub-space
defined by:
    - exactly n_alpha alpha (spin-up) electrons
    - exactly n_beta  beta  (spin-down) electrons

For a singlet ground state: n_alpha = n_beta = active_electrons // 2.

PennyLane JW spin-orbital convention (verified empirically):
    qubit index 0, 2, 4, ... -> alpha (spin-up)  orbitals
    qubit index 1, 3, 5, ... -> beta  (spin-down) orbitals
    Qubit 0 = leftmost character in the format() bit string.

Output format (.npz)
--------------------
    hamiltonian    : full 2^n x 2^n matrix (same as original; used by VQE)
    weights        : Pauli coefficients
    pauli_strings  : Pauli operator strings
    eigvals        : eigenvalues in the PHYSICAL sector (n_alpha, n_beta)
    eigvals_full   : eigenvalues of the full (unrestricted) Hamiltonian
    energy_shift   : nuclear repulsion energy
    n_electrons    : active_electrons
    n_alpha        : number of spin-up electrons in the target sector
    n_beta         : number of spin-down electrons in the target sector
    sector_indices : indices (in the full 2^n basis) of the physical sector states

e0_full   = eigvals_full[0] + energy_shift   # may correspond to wrong charge
e0_sector = eigvals[0]      + energy_shift   # physically correct ground state
"""

import pennylane as qml
from pennylane import qchem
import numpy as np
from pathlib import Path

MOL_DATA_DIR = Path(__file__).resolve().parent.parent / "mol_data_physical"
MOL_DATA_DIR.mkdir(exist_ok=True)


# ── Sector projection ─────────────────────────────────────────────────────────

def get_sector_indices(n_qubits, n_alpha, n_beta):
    """
    Return the indices (in the full 2^n_qubits computational basis) of all
    states that have exactly n_alpha alpha electrons and n_beta beta electrons.

    PennyLane JW convention:
        qubit j occupancy = int(format(i, '0{n}b')[j])   (qubit 0 = leftmost)
        even-j qubits (0, 2, 4, ...) -> alpha spin-orbitals
        odd-j  qubits (1, 3, 5, ...) -> beta  spin-orbitals
    """
    indices = []
    for i in range(2 ** n_qubits):
        bits = format(i, f'0{n_qubits}b')          # qubit 0 = bits[0]
        na = sum(int(bits[j]) for j in range(0, n_qubits, 2))   # even positions
        nb = sum(int(bits[j]) for j in range(1, n_qubits, 2))   # odd positions
        if na == n_alpha and nb == n_beta:
            indices.append(i)
    return np.array(indices, dtype=int)


# ── Main generation function ──────────────────────────────────────────────────

def generate_mol_data_physical(mol_name, symbols, coordinates_angstrom,
                               active_electrons, active_orbitals,
                               geometry_str, mapping='jordan_wigner',
                               charge=0, mult=1):
    """
    Generate .npz mol data with eigenvalues restricted to the correct
    particle-number and spin sector.

    Parameters
    ----------
    active_electrons : int
        Number of electrons in the active space.  Must be even for singlet (mult=1).
    active_orbitals  : int
        Number of spatial orbitals in the active space.
        Results in n_qubits = 2 * active_orbitals.
    mult : int
        Spin multiplicity (2S+1).  Only singlet (mult=1) is fully supported;
        for doublets the sector is n_alpha = n_beta + 1.
    """
    assert active_electrons % 2 == 0 or mult != 1, \
        "Singlet requires even number of active electrons."

    coordinates_bohr = coordinates_angstrom * 1.8897259886

    H, n_qubits = qchem.molecular_hamiltonian(
        symbols, coordinates_bohr, charge=charge, mult=mult,
        active_electrons=active_electrons, active_orbitals=active_orbitals,
        mapping=mapping
    )

    # ── Separate identity (nuclear repulsion) from electronic terms ───────────
    coeffs, ops = H.terms()
    energy_shift = 0.0
    electronic_coeffs, electronic_ops = [], []
    for coeff, op in zip(coeffs, ops):
        if op.name in ['Identity', 'I']:
            energy_shift += float(coeff)
        else:
            electronic_coeffs.append(float(coeff))
            electronic_ops.append(op)

    # ── Build full 2^n x 2^n Hamiltonian matrix ───────────────────────────────
    dim = 2 ** n_qubits
    ham_mat = np.zeros((dim, dim), dtype=np.complex128)
    for c, o in zip(electronic_coeffs, electronic_ops):
        ham_mat += c * qml.matrix(o, wire_order=range(n_qubits))

    # ── Full-space eigenvalues (original, possibly unphysical) ────────────────
    eigvals_full = np.linalg.eigh(ham_mat)[0]

    # ── Physical sector: correct particle number + spin ───────────────────────
    # For singlet: n_alpha = n_beta = n_electrons / 2
    # For doublet: n_alpha = n_beta + 1  (higher Sz = +1/2 component chosen)
    if mult == 1:
        n_alpha = active_electrons // 2
        n_beta  = active_electrons // 2
    else:
        S = (mult - 1) / 2
        n_alpha = int(active_electrons / 2 + S)   # highest Sz sector
        n_beta  = int(active_electrons / 2 - S)

    sector_indices = get_sector_indices(n_qubits, n_alpha, n_beta)
    H_sector = ham_mat[np.ix_(sector_indices, sector_indices)]
    eigvals_sector = np.linalg.eigh(np.real(H_sector))[0]   # sector H is real

    # ── Print summary ─────────────────────────────────────────────────────────
    e0_full   = eigvals_full[0]   + energy_shift
    e0_sector = eigvals_sector[0] + energy_shift
    print(f"[OK] {mol_name}  ({n_qubits}q, ae={active_electrons}, ao={active_orbitals})")
    print(f"     sector ({n_alpha}α + {n_beta}β),  {len(sector_indices)} states")
    print(f"     E0_full   = {e0_full:.8f}   (may be wrong sector)")
    print(f"     E0_sector = {e0_sector:.8f}   (physically correct)")
    if abs(e0_full - e0_sector) > 1e-6:
        print(f"     *** DIFFERENCE = {e0_full - e0_sector:.6f} Ha  <-- sector matters here!")
    print()

    # ── Save ──────────────────────────────────────────────────────────────────
    processed_geometry = geometry_str.replace(" ", "_")
    filename = MOL_DATA_DIR / f"{mol_name}_{n_qubits}q_geom_{processed_geometry}_{mapping}.npz"

    np.savez(filename,
             hamiltonian    = ham_mat,
             weights        = np.array(electronic_coeffs),
             pauli_strings  = np.array([str(op) for op in electronic_ops]),
             eigvals        = eigvals_sector,   # physically correct (sector)
             eigvals_full   = eigvals_full,      # full space (for comparison)
             energy_shift   = energy_shift,
             n_electrons    = active_electrons,
             n_alpha        = n_alpha,
             n_beta         = n_beta,
             sector_indices = sector_indices)

    return e0_sector


# ── Molecule list (mirrors prepare_molecules.py exactly) ──────────────────────

# --- L1: Minimalism ---
e = generate_mol_data_physical(
    "L1_H2_Equil", ["H", "H"], np.array([[0,0,0],[0,0,0.735]]),
    active_electrons=2, active_orbitals=2,
    geometry_str="H .0 .0 0.0; H .0 .0 0.735"
)

e = generate_mol_data_physical(
    "L1_BH", ["B", "H"], np.array([[0,0,0],[0,0,1.232]]),
    active_electrons=2, active_orbitals=3,
    geometry_str="B .0 .0 0.0; H .0 .0 1.232"
)

# --- L2: Asymmetry ---
e = generate_mol_data_physical(
    "L2_BeH_Plus", ["Be", "H"], np.array([[0,0,0],[0,0,1.312]]),
    active_electrons=2, active_orbitals=2,
    geometry_str="Be .0 .0 0.0; H .0 .0 1.312", charge=1
)

e = generate_mol_data_physical(
    "L2_LiH_Equil", ["Li", "H"], np.array([[0,0,0],[0,0,1.595]]),
    active_electrons=2, active_orbitals=3,
    geometry_str="Li .0 .0 0.0; H .0 .0 1.595"
)

e = generate_mol_data_physical(
    "L2_BF", ["B", "F"], np.array([[0,0,0],[0,0,1.267]]),
    active_electrons=6, active_orbitals=4,
    geometry_str="B .0 .0 0.0; F .0 .0 1.267"
)

# --- L3: Stability ---
e = generate_mol_data_physical(
    "L3_HeH_Plus", ["He", "H"], np.array([[0,0,0],[0,0,0.774]]),
    active_electrons=2, active_orbitals=2,
    geometry_str="He .0 .0 0.0; H .0 .0 0.774", charge=1
)

e = generate_mol_data_physical(
    "L3_CH2_Singlet", ["C", "H", "H"], np.array([[0,0,0],[0,0.86,0.73],[0,-0.86,0.73]]),
    active_electrons=2, active_orbitals=3,
    geometry_str="C .0 .0 0.0; H .0 0.86 0.73; H .0 -0.86 0.73"
)

e = generate_mol_data_physical(
    "L3_LiH_Stretch", ["Li", "H"], np.array([[0,0,0],[0,0,3.500]]),
    active_electrons=2, active_orbitals=3,
    geometry_str="Li .0 .0 0.0; H .0 .0 3.500"
)

e = generate_mol_data_physical(
    "L3_H3_Triangle", ["H", "H", "H"], np.array([[0,0,0],[1,0,0],[0.5,0.866,0]]),
    active_electrons=2, active_orbitals=3,
    geometry_str="H .0 .0 0.0; H 1.0 .0 0.0; H 0.5 0.866 0.0", charge=1
)

# --- L4: Representation ---
e = generate_mol_data_physical(
    "L4_H2_Stretch", ["H", "H"], np.array([[0,0,0],[0,0,2.5]]),
    active_electrons=2, active_orbitals=2,
    geometry_str="H .0 .0 0.0; H .0 .0 2.5"
)

e = generate_mol_data_physical(
    "L4_H3_Linear", ["H", "H", "H"], np.array([[0,0,0],[0,0,1],[0,0,2]]),
    active_electrons=2, active_orbitals=3,
    geometry_str="H .0 .0 0.0; H .0 .0 1.0; H .0 .0 2.0", charge=1
)

e = generate_mol_data_physical(
    "L4_H2O_StrongCorr", ["O", "H", "H"], np.array([[0,0,0],[0,0.757,0.586],[0,-0.757,0.586]]),
    active_electrons=4, active_orbitals=4,
    geometry_str="O .0 .0 0.0; H .0 0.757 0.586; H .0 -0.757 0.586"
)

# --- L5: Topology ---
e = generate_mol_data_physical(
    "L5_H3_Linear", ["H", "H", "H"], np.array([[0,0,0],[0,0,1],[0,0,2]]),
    active_electrons=2, active_orbitals=3,
    geometry_str="H .0 .0 0.0; H .0 .0 1.0; H .0 .0 2.0", charge=1
)

e = generate_mol_data_physical(
    "L5_H4_Chain", ["H", "H", "H", "H"], np.array([[0,0,0],[0,0,1],[0,0,2],[0,0,3]]),
    active_electrons=4, active_orbitals=4,
    geometry_str="H .0 .0 0.0; H .0 .0 1.0; H .0 .0 2.0; H .0 .0 3.0"
)

# --- L6: Scalability ---
e = generate_mol_data_physical(
    "L6_BeH2_Scalability", ["Be", "H", "H"], np.array([[0,0,0],[0,0,1.326],[0,0,-1.326]]),
    active_electrons=2, active_orbitals=5,
    geometry_str="Be .0 .0 0.0; H .0 .0 1.326; H .0 .0 -1.326"
)
