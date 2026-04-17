"""Generate an 8-qubit CH2 singlet benchmark candidate.

This variant keeps the original Level-3 CH2 geometry but enlarges the active
space from CAS(2e,3o) to CAS(2e,4o), which maps to 8 qubits under
Jordan-Wigner. The goal is to test whether the current CH2 candidate becomes
meaningfully harder once the active-space truncation is relaxed.

Usage:
  conda run -n crlqas_env python mol_gen/prepare_ch2_l3_8q.py
"""

from __future__ import annotations

import numpy as np

from prepare_molecules import generate_mol_data_for_rl
from prepare_molecules_physical import generate_mol_data_physical


MOL_NAME = "L3_CH2_Singlet"
SYMBOLS = ["C", "H", "H"]
COORDS = np.array(
    [
        [0.0, 0.0, 0.0],
        [0.0, 0.86, 0.73],
        [0.0, -0.86, 0.73],
    ],
    dtype=float,
)
GEOMETRY = "C .0 .0 0.0; H .0 0.86 0.73; H .0 -0.86 0.73"
ACTIVE_ELECTRONS = 2
ACTIVE_ORBITALS = 4
MAPPING = "jordan_wigner"


def main():
    print("=== L3 CH2 singlet 8q active-space expansion ===")
    print(f"active electrons = {ACTIVE_ELECTRONS}")
    print(f"active orbitals  = {ACTIVE_ORBITALS}")
    print(f"target qubits    = {2 * ACTIVE_ORBITALS}")
    print(f"geometry         = {GEOMETRY}")
    print()

    full_e0 = generate_mol_data_for_rl(
        mol_name=MOL_NAME,
        symbols=SYMBOLS,
        coordinates_angstrom=COORDS,
        active_electrons=ACTIVE_ELECTRONS,
        active_orbitals=ACTIVE_ORBITALS,
        geometry_str=GEOMETRY,
        mapping=MAPPING,
    )

    sector_e0 = generate_mol_data_physical(
        mol_name=MOL_NAME,
        symbols=SYMBOLS,
        coordinates_angstrom=COORDS,
        active_electrons=ACTIVE_ELECTRONS,
        active_orbitals=ACTIVE_ORBITALS,
        geometry_str=GEOMETRY,
        mapping=MAPPING,
    )

    print()
    print("=== Summary ===")
    print(f"full-space ground energy   = {full_e0:.8f}")
    print(f"physical-sector energy     = {sector_e0:.8f}")


if __name__ == "__main__":
    main()
