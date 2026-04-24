"""
Exact-ground-state entanglement fingerprints for benchmark molecules.

This script computes a simple Level-4-style entanglement diagnostic directly
from the exact ground state:

  1. Diagonalize the molecular Hamiltonian (full-space or physical-sector)
  2. Obtain the exact ground-state wavefunction |psi_0>
  3. For each qubit q, trace out all other qubits
  4. Compute the single-qubit Von Neumann entropy

The entropy is reported in bits:

    S_q = -Tr(rho_q log2 rho_q),    0 <= S_q <= 1

Interpretation:
  - S_q ~ 0 : qubit q is weakly entangled with the rest
  - S_q ~ 1 : qubit q is strongly entangled with the rest

Important caveat:
  If the ground state is exactly degenerate or nearly degenerate, the chosen
  eigenvector inside that low-energy subspace may not be unique, so the
  resulting entropies should be interpreted with care.

Usage examples:
  python mol_gen/read_entanglement_fingerprints.py
  python mol_gen/read_entanglement_fingerprints.py --group fixed
  python mol_gen/read_entanglement_fingerprints.py --source physical
  python mol_gen/read_entanglement_fingerprints.py --group l4 --source physical
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh

np.set_printoptions(precision=4, suppress=True, linewidth=160)

ROOT = Path(__file__).resolve().parent.parent
FULL_DIR = ROOT / "mol_data"
PHYS_DIR = ROOT / "mol_data_physical"

_DENSE_MAX_DIM = 1024
_DEGEN_TOL = 1e-8


GROUPS = {
    "l2": [
        ("LiH_Equil", "L2_LiH_Equil_6q_geom_Li_.0_.0_0.0;_H_.0_.0_1.595_jordan_wigner.npz"),
        ("BF", "L2_BF_8q_geom_B_.0_.0_0.0;_F_.0_.0_1.267_jordan_wigner.npz"),
        ("BeH_plus", "L2_BeHplus_4q_geom_Be_.0_.0_0.0;_H_.0_.0_1.343_jordan_wigner.npz"),
    ],
    "l4": [
        ("H2_Stretch", "L4_H2_Stretch_4q_geom_H_.0_.0_0.0;_H_.0_.0_2.5_jordan_wigner.npz"),
        ("H3_Linear", "L4_H3_Linear_6q_geom_H_.0_.0_0.0;_H_.0_.0_1.0;_H_.0_.0_2.0_jordan_wigner.npz"),
        ("H2O", "L4_H2O_StrongCorr_8q_geom_O_.0_.0_0.0;_H_.0_1.186_0.918;_H_.0_-1.186_0.918_jordan_wigner.npz"),
    ],
    "fixed": [
        ("L1_BeH2_STO3G", "L1_BeH2_STO3G_6q_geom_Be_.0_.0_0.0;_H_.0_.0_1.326;_H_.0_.0_-1.326_jordan_wigner.npz"),
        ("L2_LiH_Equil",  "L2_LiH_Equil_6q_geom_Li_.0_.0_0.0;_H_.0_.0_1.595_jordan_wigner.npz"),
        ("L3_CH2_Singlet", "L3_CH2_Singlet_8q_geom_C_.0_.0_0.0;_H_.0_0.86_0.73;_H_.0_-0.86_0.73_jordan_wigner.npz"),
        ("L4_H2O",        "L4_H2O_StrongCorr_8q_geom_O_.0_.0_0.0;_H_.0_1.186_0.918;_H_.0_-1.186_0.918_jordan_wigner.npz"),
        ("L5_H3_Linear",  "L5_H3_Linear_6q_geom_H_.0_.0_0.0;_H_.0_.0_1.0;_H_.0_.0_2.0_jordan_wigner.npz"),
        ("L6_BeH2_631G",  "L6_BeH2_631G_8q_geom_Be_.0_.0_0.0;_H_.0_.0_1.326;_H_.0_.0_-1.326_jordan_wigner.npz"),
    ],
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute exact-ground-state single-qubit Von Neumann entropies."
    )
    parser.add_argument(
        "--group",
        default="l4",
        choices=sorted(GROUPS.keys()),
        help="Which predefined molecule group to analyze.",
    )
    parser.add_argument(
        "--source",
        default="full",
        choices=["full", "physical"],
        help="Use unrestricted full-space files or physical-sector files.",
    )
    parser.add_argument(
        "--degeneracy_tol",
        type=float,
        default=_DEGEN_TOL,
        help="Tolerance for flagging near-degeneracy in the lowest spectrum.",
    )
    return parser.parse_args()


def _smallest_eigpair(H: np.ndarray):
    """
    Return a few lowest eigenpairs for a Hermitian Hamiltonian.

    Dense diagonalization is used for smaller matrices. For larger matrices,
    sparse eigsh is used to avoid materializing the full eigensystem.
    """
    dim = H.shape[0]
    if dim <= _DENSE_MAX_DIM:
        eigvals, eigvecs = np.linalg.eigh(H)
        return eigvals, eigvecs

    k = min(6, dim - 2)
    evals, evecs = eigsh(sp.csr_matrix(H), k=k, which="SA")
    order = np.argsort(evals.real)
    evals = np.asarray(evals.real[order])
    evecs = np.asarray(evecs[:, order])
    return evals, evecs


def _single_qubit_entropies(state: np.ndarray, n_qubits: int):
    """Return S_q for each qubit q, using log base 2."""
    psi = np.asarray(state, dtype=np.complex128)
    psi = psi / np.linalg.norm(psi)
    tensor = psi.reshape((2,) * n_qubits)

    entropies = []
    for q in range(n_qubits):
        psi_q = np.moveaxis(tensor, q, 0).reshape(2, -1)
        rho_q = psi_q @ psi_q.conj().T
        evals = np.linalg.eigvalsh(rho_q)
        evals = np.clip(evals.real, 0.0, 1.0)
        mask = evals > 1e-14
        sq = float(-np.sum(evals[mask] * np.log2(evals[mask])))
        entropies.append(sq)
    return np.array(entropies, dtype=float)


def _load_ground_state(npz_path: Path, source: str):
    """
    Return exact ground-state vector embedded in the full 2^n computational basis.

    For source='full', the ground state is taken from the full Hamiltonian.
    For source='physical', the ground state is taken from H_sector and then
    embedded back into the full Hilbert space using sector_indices.
    """
    data = np.load(npz_path, allow_pickle=True)
    H_full = np.asarray(data["hamiltonian"], dtype=np.complex128)
    n_qubits = int(np.log2(H_full.shape[0]))
    energy_shift = float(data.get("energy_shift", 0.0))

    if source == "full":
        eigvals, eigvecs = _smallest_eigpair(H_full)
        gs = eigvecs[:, 0]
        return dict(
            state=gs,
            n_qubits=n_qubits,
            e0=float(eigvals[0] + energy_shift),
            gap01=(float(eigvals[1] - eigvals[0]) if len(eigvals) > 1 else float("nan")),
            degeneracy_count=int(np.sum(np.abs(eigvals - eigvals[0]) <= _DEGEN_TOL)),
            sector_size=None,
        )

    sector_idx = data["sector_indices"].astype(int)
    H_sector = H_full[np.ix_(sector_idx, sector_idx)]
    eigvals, eigvecs = _smallest_eigpair(np.real(H_sector))
    gs_sector = eigvecs[:, 0]
    gs_full = np.zeros(H_full.shape[0], dtype=np.complex128)
    gs_full[sector_idx] = gs_sector
    return dict(
        state=gs_full,
        n_qubits=n_qubits,
        e0=float(eigvals[0] + energy_shift),
        gap01=(float(eigvals[1] - eigvals[0]) if len(eigvals) > 1 else float("nan")),
        degeneracy_count=int(np.sum(np.abs(eigvals - eigvals[0]) <= _DEGEN_TOL)),
        sector_size=len(sector_idx),
    )


def inspect_molecule(label: str, filename: str, source: str, degeneracy_tol: float):
    directory = FULL_DIR if source == "full" else PHYS_DIR
    npz_path = directory / filename
    if not npz_path.exists():
        print(f"[SKIP] {label:<14} missing: {npz_path.name}")
        return None

    global _DEGEN_TOL
    _DEGEN_TOL = degeneracy_tol
    info = _load_ground_state(npz_path, source=source)
    ent = _single_qubit_entropies(info["state"], info["n_qubits"])

    note = ""
    if info["degeneracy_count"] >= 2:
        note = "  [near/exact GS degeneracy: entropy may depend on chosen eigenvector]"

    print(f"\n{'=' * 72}")
    print(f"  {label}  ({info['n_qubits']} qubits, source={source}){note}")
    print(f"{'=' * 72}")
    if info["sector_size"] is not None:
        print(f"  sector size      = {info['sector_size']}")
    print(f"  E0               = {info['e0']:.8f}")
    print(f"  Gap01            = {info['gap01']:.8f}")
    print(f"  GS multiplicity* = {info['degeneracy_count']}  (tol={degeneracy_tol:g})")
    print(f"  S(q)             = {np.array2string(ent, precision=4, separator=', ')}")
    print(f"  mean S(q)        = {ent.mean():.6f}")
    print(f"  max  S(q)        = {ent.max():.6f}")
    print(f"  min  S(q)        = {ent.min():.6f}")
    print(f"  std  S(q)        = {ent.std():.6f}")
    print(f"  argmax qubit     = {int(np.argmax(ent))}")

    return dict(
        molecule=label,
        qubits=info["n_qubits"],
        source=source,
        e0=info["e0"],
        gap01=info["gap01"],
        gs_multiplicity=info["degeneracy_count"],
        mean_s1=float(ent.mean()),
        max_s1=float(ent.max()),
        min_s1=float(ent.min()),
        std_s1=float(ent.std()),
        argmax_q=int(np.argmax(ent)),
    )


def _fmt(v):
    if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
        return "   N/A"
    return f"{v:7.4f}"


def print_summary(rows):
    if not rows:
        return
    print(f"\n{'=' * 110}")
    print("  SUMMARY")
    print(f"{'=' * 110}")
    print(
        f"  {'Molecule':<16} {'q':>2} {'src':>8} {'Gap01':>10} {'mult':>6} "
        f"{'mean S1':>9} {'max S1':>9} {'min S1':>9} {'std S1':>9} {'argmax':>7}"
    )
    print("  " + "-" * 108)
    for r in rows:
        print(
            f"  {r['molecule']:<16} {r['qubits']:>2} {r['source']:>8} "
            f"{r['gap01']:>10.6f} {r['gs_multiplicity']:>6} "
            f"{_fmt(r['mean_s1'])} {_fmt(r['max_s1'])} {_fmt(r['min_s1'])} "
            f"{_fmt(r['std_s1'])} {r['argmax_q']:>7}"
        )


def main():
    args = parse_args()
    rows = []
    for label, filename in GROUPS[args.group]:
        row = inspect_molecule(label, filename, args.source, args.degeneracy_tol)
        if row is not None:
            rows.append(row)
    print_summary(rows)


if __name__ == "__main__":
    main()
