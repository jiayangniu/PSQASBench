"""Generate geometry variants of the L3 CH2 singlet while keeping 6 qubits.

This helper exists because the current `L3_CH2_Singlet_6q` looks too easy in
the benchmark.  Before replacing the molecule family entirely, we first try a
small set of geometry distortions that keep the active space fixed at:

  - active_electrons = 2
  - active_orbitals = 3
  - n_qubits = 6

Each case is generated into both:
  - `mol_data/`            (full-space eigvals)
  - `mol_data_physical/`   (sector-restricted eigvals)

Usage:
  conda run -n crlqas_env python mol_gen/prepare_ch2_l3_variants.py
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from prepare_molecules import generate_mol_data_for_rl
from prepare_molecules_physical import generate_mol_data_physical


ACTIVE_ELECTRONS = 2
ACTIVE_ORBITALS = 3
MAPPING = "jordan_wigner"
SYMBOLS = ["C", "H", "H"]


def bent_ch2_geometry(bond_length_angstrom: float, angle_deg: float) -> tuple[np.ndarray, str]:
    """Return a symmetric bent-CH2 geometry and its filename-friendly string.

    Carbon stays at the origin, and the two hydrogens are placed symmetrically
    in the yz-plane so that the H-C-H angle equals `angle_deg`.
    """
    half = math.radians(angle_deg / 2.0)
    y = bond_length_angstrom * math.sin(half)
    z = bond_length_angstrom * math.cos(half)
    coords = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, +y, z],
            [0.0, -y, z],
        ],
        dtype=float,
    )
    geom = (
        f"C .0 .0 0.0; "
        f"H .0 {y:.3f} {z:.3f}; "
        f"H .0 {-y:.3f} {z:.3f}"
    )
    return coords, geom


CASES = [
    # More compressed than the current ~99 deg baseline.
    ("L3_CH2_Singlet_R113_A080", 1.13, 80.0),
    # Wider angle at roughly the same bond length.
    ("L3_CH2_Singlet_R113_A130", 1.13, 130.0),
    # Stretched bond near the current angle regime.
    ("L3_CH2_Singlet_R130_A100", 1.30, 100.0),
    # Joint stretch + wide-angle distortion.
    ("L3_CH2_Singlet_R130_A130", 1.30, 130.0),
]


def main():
    print("=== L3 CH2 singlet geometry sweep (fixed 6q active space) ===")
    print(f"active electrons = {ACTIVE_ELECTRONS}")
    print(f"active orbitals  = {ACTIVE_ORBITALS}")
    print()

    results = []
    for mol_name, bond_length, angle_deg in CASES:
        print(f"[CASE] {mol_name}: bond={bond_length:.2f} A, angle={angle_deg:.1f} deg")
        coords, geom = bent_ch2_geometry(bond_length, angle_deg)

        full_e0 = generate_mol_data_for_rl(
            mol_name=mol_name,
            symbols=SYMBOLS,
            coordinates_angstrom=coords,
            active_electrons=ACTIVE_ELECTRONS,
            active_orbitals=ACTIVE_ORBITALS,
            geometry_str=geom,
            mapping=MAPPING,
        )

        generate_mol_data_physical(
            mol_name=mol_name,
            symbols=SYMBOLS,
            coordinates_angstrom=coords,
            active_electrons=ACTIVE_ELECTRONS,
            active_orbitals=ACTIVE_ORBITALS,
            geometry_str=geom,
            mapping=MAPPING,
        )

        results.append(
            {
                "name": mol_name,
                "bond_length": bond_length,
                "angle_deg": angle_deg,
                "geometry": geom,
                "full_ground_energy": full_e0,
            }
        )

    print()
    print("=== Summary ===")
    for item in results:
        print(
            f"{item['name']:26s}  "
            f"bond={item['bond_length']:.2f}A  "
            f"angle={item['angle_deg']:>5.1f}deg  "
            f"E0_full={item['full_ground_energy']:.8f}"
        )
        print(f"  geometry = {item['geometry']}")


if __name__ == "__main__":
    main()
