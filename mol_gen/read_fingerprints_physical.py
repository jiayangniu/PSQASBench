"""
Hamiltonian Fingerprint Analysis for QAS Benchmarking (physical-sector version).

This script reads `mol_data_physical/` and computes fingerprint metrics on the
projected physical-sector Hamiltonian H_sector defined by the stored
particle-number / spin-sector indices.

Pauli-decomposition metrics (hub score, Z-ratio, high-order ratio, etc.) are
still computed from the original Pauli expansion stored in the file.  The
Gershgorin metrics G1-G4 and the printed energy gap are computed on H_sector.

Usage:
  python read_fingerprints_physical.py
"""

import numpy as np
from collections import defaultdict
from pathlib import Path

np.set_printoptions(precision=4, suppress=True, linewidth=160)

MOL_DATA_DIR = Path(__file__).resolve().parent.parent / "mol_data_physical"


def _wire_indices(op_str: str):
    """Return list of qubit indices appearing in a Pauli string like 'Z(0)@X(2)'."""
    wires, i = [], 0
    while i < len(op_str):
        if op_str[i] == "(":
            j = op_str.index(")", i)
            wires.append(int(op_str[i + 1:j]))
            i = j
        i += 1
    return wires


def _pauli_category(op_str: str):
    """Classify a Pauli string as Z-only, XY-only, mixed, or identity."""
    has_z = "Z(" in op_str
    has_xy = "X(" in op_str or "Y(" in op_str
    if not has_z and not has_xy:
        return "identity"
    if has_z and not has_xy:
        return "Z-only"
    if has_xy and not has_z:
        return "XY-only"
    return "mixed"


def compute_pauli_metrics(weights, pauli_strings, n_qubits):
    """Compute fingerprint metrics from the Pauli decomposition."""
    if pauli_strings is None:
        nan = float("nan")
        return dict(
            hub_score=nan,
            asymmetry=nan,
            q_importance=np.full(n_qubits, nan),
            z_ratio=nan,
            xy_ratio=nan,
            mixed_ratio=nan,
            high_order_ratio=nan,
        )

    q_imp = np.zeros(n_qubits)
    w_total = 0.0
    w_z = w_xy = w_mixed = 0.0
    w_by_body = defaultdict(float)

    for coeff, op in zip(weights, pauli_strings):
        w = abs(coeff)
        w_total += w
        for q in _wire_indices(str(op)):
            q_imp[q] += w
        cat = _pauli_category(str(op))
        body = sum(ch in "XYZ" for ch in str(op))
        if cat == "Z-only":
            w_z += w
        elif cat == "XY-only":
            w_xy += w
        elif cat == "mixed":
            w_mixed += w
        w_by_body[body] += w

    mean_imp = q_imp.mean()
    hub_score = q_imp.max() / mean_imp if mean_imp > 0 else 0.0
    asymmetry = (q_imp.max() - q_imp.min()) / mean_imp if mean_imp > 0 else 0.0

    return dict(
        hub_score=hub_score,
        asymmetry=asymmetry,
        q_importance=q_imp,
        z_ratio=w_z / w_total,
        xy_ratio=w_xy / w_total,
        mixed_ratio=w_mixed / w_total,
        high_order_ratio=sum(v for k, v in w_by_body.items() if k >= 4) / w_total,
    )


def compute_g1(H):
    diag = np.real(np.diag(H))
    radii = np.sum(np.abs(H), axis=1) - np.abs(diag)
    return float(np.sum(np.abs(diag) >= radii)) / H.shape[0]


def compute_g2(H):
    diag = np.real(np.diag(H))
    radii = np.sum(np.abs(H), axis=1) - np.abs(diag)
    mask = radii > 0
    if not np.any(mask):
        return float("inf")
    return float(np.min(np.abs(diag[mask]) / radii[mask]))


def compute_g3(H):
    diag = np.real(np.diag(H))
    radii = np.sum(np.abs(H), axis=1) - np.abs(diag)
    i_star = int(np.argmin(diag))
    r = radii[i_star]
    if r < 1e-14:
        return float("inf")
    return float(np.abs(diag[i_star]) / r)


def compute_g4(H):
    diag = np.real(np.diag(H))
    radii = np.sum(np.abs(H), axis=1) - np.abs(diag)
    i_star = int(np.argmin(diag))
    upper_gs = diag[i_star] + radii[i_star]
    mask_others = np.ones(len(diag), dtype=bool)
    mask_others[i_star] = False
    lower_others = float(np.min(diag[mask_others] - radii[mask_others]))
    return float(lower_others - upper_gs)


def inspect_molecule(mol_name, npz_path, charge=0):
    if not npz_path.exists():
        print(f"[SKIP] {npz_path.name}")
        return None

    data = np.load(npz_path, allow_pickle=True)
    H_full = data["hamiltonian"]
    sector_idx = data["sector_indices"].astype(int)
    H = np.real(H_full[np.ix_(sector_idx, sector_idx)])
    weights = data["weights"]
    eigvals = data["eigvals"]
    energy_shift = float(data.get("energy_shift", 0.0))
    n_qubits = int(np.log2(H_full.shape[0]))
    pauli_strings = data["pauli_strings"] if "pauli_strings" in data else None
    n_alpha = int(data["n_alpha"])
    n_beta = int(data["n_beta"])

    pm = compute_pauli_metrics(weights, pauli_strings, n_qubits)
    g1 = compute_g1(H)
    g2 = compute_g2(H)
    g3 = compute_g3(H)
    g4 = compute_g4(H)
    gap = eigvals[1] - eigvals[0]
    top5 = eigvals[:5] + energy_shift

    miss = "  [! no pauli_strings]" if pauli_strings is None else ""
    charge_tag = f"  charge={charge:+d}" if charge != 0 else ""
    print(f"\n{'='*60}")
    print(
        f"  {mol_name}  ({n_qubits} qubits){charge_tag}"
        f"  sector=({n_alpha}a+{n_beta}b){miss}"
    )
    print(f"{'='*60}")

    def f(v):
        return f"{v:.4f}" if not (isinstance(v, float) and np.isnan(v)) else "N/A"

    def fg(v):
        return f"{v:.4f}" if np.isfinite(v) else "  inf"

    print(f"  [1] Energy gap       Gap01        = {gap:.6f}")
    print(
        f"  [2] Qubit bias       Hub Score    = {f(pm['hub_score'])}"
        f"    Asymmetry = {f(pm['asymmetry'])}"
    )
    if pauli_strings is not None:
        print(f"      Qubit importance             = {pm['q_importance']}")
    print(
        f"  [3] Pauli types      Z-only       = {f(pm['z_ratio'])}"
        f"    XY-only = {f(pm['xy_ratio'])}"
        f"    Mixed = {f(pm['mixed_ratio'])}"
    )
    print(f"  [4] High-order ratio (>=4-body)   = {f(pm['high_order_ratio'])}")
    print(f"  [5] Top-5 physical-sector energies (with shift):")
    for i, e in enumerate(top5):
        print(f"        E{i} = {e:.8f}")
    print(f"  [G1] Diag-dom frac   G1           = {g1:.4f}")
    print(f"  [G2] Worst-case DD   G2           = {fg(g2)}")
    print(f"  [G3] HF-state DD     G3           = {fg(g3)}")
    print(f"  [G4] GS disc sep     G4           = {g4:.4f}")

    return dict(
        molecule=mol_name,
        charge=charge,
        qubits=n_qubits,
        gap01=gap,
        hub_score=pm["hub_score"],
        asymmetry=pm["asymmetry"],
        z_ratio=pm["z_ratio"],
        xy_ratio=pm["xy_ratio"],
        mixed_ratio=pm["mixed_ratio"],
        high_order=pm["high_order_ratio"],
        g1=g1,
        g2=g2,
        g3=g3,
        g4=g4,
    )


MOLECULES = [
    ("H2_Equil", "L1_H2_Equil_4q_geom_H_.0_.0_0.0;_H_.0_.0_0.735_jordan_wigner.npz", 0),
    ("BH", "L1_BH_6q_geom_B_.0_.0_0.0;_H_.0_.0_1.232_jordan_wigner.npz", 0),
    ("BeH_Plus", "L2_BeH_Plus_4q_geom_Be_.0_.0_0.0;_H_.0_.0_1.312_jordan_wigner.npz", +1),
    ("LiH_Equil", "L2_LiH_Equil_6q_geom_Li_.0_.0_0.0;_H_.0_.0_1.595_jordan_wigner.npz", 0),
    ("BF", "L2_BF_8q_geom_B_.0_.0_0.0;_F_.0_.0_1.267_jordan_wigner.npz", 0),
    ("HeH_Plus", "L3_HeH_Plus_4q_geom_He_.0_.0_0.0;_H_.0_.0_0.774_jordan_wigner.npz", +1),
    ("CH2_Singlet", "L3_CH2_Singlet_6q_geom_C_.0_.0_0.0;_H_.0_0.86_0.73;_H_.0_-0.86_0.73_jordan_wigner.npz", 0),
    ("LiH_Stretch", "L3_LiH_Stretch_6q_geom_Li_.0_.0_0.0;_H_.0_.0_3.500_jordan_wigner.npz", 0),
    ("H3_Triangle", "L3_H3_Triangle_6q_geom_H_.0_.0_0.0;_H_1.0_.0_0.0;_H_0.5_0.866_0.0_jordan_wigner.npz", +1),
    ("H2_Stretch", "L4_H2_Stretch_4q_geom_H_.0_.0_0.0;_H_.0_.0_2.5_jordan_wigner.npz", 0),
    ("H3_Linear", "L4_H3_Linear_6q_geom_H_.0_.0_0.0;_H_.0_.0_1.0;_H_.0_.0_2.0_jordan_wigner.npz", +1),
    ("H2O", "L4_H2O_StrongCorr_8q_geom_O_.0_.0_0.0;_H_.0_0.757_0.586;_H_.0_-0.757_0.586_jordan_wigner.npz", 0),
    ("H4_Chain", "L5_H4_Chain_8q_geom_H_.0_.0_0.0;_H_.0_.0_1.0;_H_.0_.0_2.0;_H_.0_.0_3.0_jordan_wigner.npz", 0),
    ("BeH2", "L6_BeH2_Scalability_10q_geom_Be_.0_.0_0.0;_H_.0_.0_1.326;_H_.0_.0_-1.326_jordan_wigner.npz", 0),
]


def _fmt(v):
    if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
        return "   inf" if np.isinf(v) else "   N/A"
    return f"{v:6.3f}"


def _print_summary(results, title):
    print(f"\n{'='*85}")
    print(f"  {title}")
    print(f"{'='*85}")
    print(
        f"  {'Molecule':<14} {'q':>2}  {'Gap01':>8}  {'Z-ratio':>7}  "
        f"{'HighOrd':>7}  {'G1':>6}  {'G2':>6}  {'G3':>6}  {'G4':>7}"
    )
    print("  " + "-" * 83)
    for r in results:
        print(
            f"  {r['molecule']:<14} {r['qubits']:>2}  {r['gap01']:>8.4f}"
            f"  {_fmt(r['z_ratio'])}  {_fmt(r['high_order'])}"
            f"  {_fmt(r['g1'])}  {_fmt(r['g2'])}  {_fmt(r['g3'])}  {r['g4']:7.3f}"
        )


neutral_results = []
charged_results = []

print("\n" + "#" * 70)
print("  NEUTRAL MOLECULES (charge = 0)  — PHYSICAL-SECTOR fingerprints")
print("#" * 70)
for name, fname, charge in MOLECULES:
    if charge == 0:
        result = inspect_molecule(name, MOL_DATA_DIR / fname, charge=charge)
        if result:
            neutral_results.append(result)

print("\n" + "#" * 70)
print("  CHARGED MOLECULES (charge != 0) — PHYSICAL-SECTOR fingerprints")
print("#" * 70)
for name, fname, charge in MOLECULES:
    if charge != 0:
        result = inspect_molecule(name, MOL_DATA_DIR / fname, charge=charge)
        if result:
            charged_results.append(result)

_print_summary(neutral_results, "SUMMARY — NEUTRAL  (physical sector)")
_print_summary(charged_results, "SUMMARY — CHARGED  (physical sector)")

print(f"\n{'='*85}")
print("  G3 STABILITY CHECK (physical sector)")
print(f"{'='*85}")
print("  Rows with G3=inf indicate the min-diagonal basis state has R_i ~ 0")
print("  (purely diagonal row — no off-diagonal coupling — G3 uninformative)\n")
for r in neutral_results + charged_results:
    status = "*** UNSTABLE (inf)" if np.isinf(r["g3"]) else f"{r['g3']:.4f}"
    print(f"  {r['molecule']:<14}  G3 = {status}")
