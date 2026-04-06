"""
Hamiltonian Fingerprint Analysis for QAS Benchmarking (full-space version).

For each molecule we compute:
  Original metrics (from Pauli decomposition):
    [1] Energy gap          Gap01 = E1 - E0
    [2] Qubit importance    I_q = sum_{i: q in P_i} |c_i|
        → Hub Score  = max(I_q) / mean(I_q)
        → Asymmetry  = (max(I_q) - min(I_q)) / mean(I_q)
    [3] Pauli-type ratios   Z-only / XY-only / Mixed (weight fractions)
    [4] High-order ratio    weight fraction of terms with >= 4 Pauli operators

  Gershgorin metrics (from Hamiltonian matrix, proposed by Akib):
    For each row i:  R_i = sum_{j != i} |H_{ij}|  (Gershgorin radius)

    [G1] Diagonal-dominance fraction
         G1 = (# rows with |h_ii| >= R_i) / (total rows)
         G1=1.0 -> fully diagonally dominant (near-diagonal H)
         G1~0   -> off-diagonal elements dominate in many rows

    [G2] Worst-case DD ratio
         G2 = min_i { |h_ii| / R_i }  (over rows with R_i > 0)
         G2 > 1 -> every row DD (implies G1=1); G2 << 1 -> worst row far from DD

    [G3] HF-state diagonal-dominance ratio
         Row i* with minimum diagonal element (proxy for Hartree-Fock state)
         G3 = |h_{i*,i*}| / R_{i*}
         NOTE: this is a single-row proxy.  The PDF discussion is closer to
         inspecting the lowest few diagonal entries / Gershgorin discs, but the
         current implementation uses only the minimum-diagonal row as a compact
         heuristic.  It is also unstable when R_{i*} ~ 0 (purely diagonal row).

    [G4] Ground-state Gershgorin disc separation
         upper(GS disc)  = h_{i*,i*} + R_{i*}  (using GS-proxy row i*)
         lower(other)    = min_{i != i*}(h_{ii} - R_i)
         G4 = lower(other) - upper(GS disc)
         G4 > 0 -> GS disc is isolated below all others
         G4 <= 0 -> GS disc overlaps at least one other disc
         As with G3, this is the single-disc version of the idea rather than a
         full "lowest few discs" analysis, so it should be treated as a narrow
         heuristic rather than a complete structural diagnostic.

This script reads `mol_data/` and computes metrics on the full, unrestricted
Hamiltonian matrix.  It does NOT impose particle-number / spin-sector
constraints when evaluating G1-G4 or reporting the eigenspectrum.

Molecules are split into neutral (charge=0) and charged (charge!=0) groups.
For charged systems in particular, the full-space ground state may lie in the
wrong sector, so these results should be interpreted as unrestricted baselines.

Interpretation note:
  G1/G2 are global diagonal-dominance summaries and align closely with the
  collaborator's diagonal-dominance / Gershgorin motivation.
  G3/G4 are heuristic operationalizations of that idea for quick comparison
  across molecules.  They should be described as proxies rather than standard
  quantum chemistry observables or complete tests of single-reference versus
  multi-reference character.

Usage:
  python read_fingerprints.py
"""

import numpy as np
from collections import defaultdict
from pathlib import Path

np.set_printoptions(precision=4, suppress=True, linewidth=160)

MOL_DATA_DIR = Path(__file__).resolve().parent.parent / "mol_data"


# ── Helper: parse qubit indices from a PennyLane-style op string ──────────────

def _wire_indices(op_str: str):
    """Return list of qubit indices appearing in a Pauli string like 'Z(0)@X(2)'."""
    wires, i = [], 0
    while i < len(op_str):
        if op_str[i] == "(":
            j = op_str.index(")", i)
            wires.append(int(op_str[i+1:j]))
            i = j
        i += 1
    return wires


def _pauli_category(op_str: str):
    """Classify a Pauli string as Z-only, XY-only, mixed, or identity."""
    has_z = "Z(" in op_str
    has_xy = "X(" in op_str or "Y(" in op_str
    if not has_z and not has_xy:   return "identity"
    if has_z and not has_xy:       return "Z-only"
    if has_xy and not has_z:       return "XY-only"
    return "mixed"


# ── Metric computations ───────────────────────────────────────────────────────

def compute_pauli_metrics(weights, pauli_strings, n_qubits):
    """
    Compute fingerprint metrics from the Pauli decomposition.
    Returns a dict; all values are NaN if pauli_strings is None.
    """
    if pauli_strings is None:
        nan = float("nan")
        return dict(hub_score=nan, asymmetry=nan, q_importance=np.full(n_qubits, nan),
                    z_ratio=nan, xy_ratio=nan, mixed_ratio=nan, high_order_ratio=nan)

    q_imp   = np.zeros(n_qubits)
    w_total = 0.0
    w_z = w_xy = w_mixed = 0.0
    w_by_body = defaultdict(float)

    for coeff, op in zip(weights, pauli_strings):
        w = abs(coeff)
        w_total += w
        for q in _wire_indices(str(op)):
            q_imp[q] += w
        cat  = _pauli_category(str(op))
        body = sum(ch in "XYZ" for ch in str(op))
        if cat == "Z-only":  w_z     += w
        elif cat == "XY-only": w_xy  += w
        elif cat == "mixed":   w_mixed += w
        w_by_body[body] += w

    mean_imp  = q_imp.mean()
    hub_score = q_imp.max() / mean_imp if mean_imp > 0 else 0.0
    asymmetry = (q_imp.max() - q_imp.min()) / mean_imp if mean_imp > 0 else 0.0

    return dict(
        hub_score      = hub_score,
        asymmetry      = asymmetry,
        q_importance   = q_imp,
        z_ratio        = w_z      / w_total,
        xy_ratio       = w_xy     / w_total,
        mixed_ratio    = w_mixed  / w_total,
        high_order_ratio = sum(v for k, v in w_by_body.items() if k >= 4) / w_total,
    )


def compute_g1(H):
    """
    G1: Diagonal-dominance fraction of the Hamiltonian matrix.

    For each row i:
      diagonal element : d_i = H[i, i]   (real, since H is Hermitian)
      Gershgorin radius: R_i = sum_{j != i} |H[i, j]|
      row is DD        : |d_i| >= R_i

    G1 = (# of DD rows) / (total # of rows)
    """
    diag  = np.real(np.diag(H))                       # shape (dim,)
    radii = np.sum(np.abs(H), axis=1) - np.abs(diag)  # R_i for each row
    n_dominant = np.sum(np.abs(diag) >= radii)
    return float(n_dominant) / H.shape[0]


def compute_g2(H):
    """
    G2: Worst-case diagonal-dominance ratio across all rows.

    For each row i, compute the ratio |h_ii| / R_i.
    G2 = min over all rows of this ratio.

    Interpretation:
      G2 > 1  : every row is diagonally dominant (implies G1 = 1.0)
      G2 < 1  : at least one row violates diagonal dominance;
                 G2 tells us how badly the worst row violates it
      G2 << 1 : the worst row is far from diagonal dominant —
                 strong off-diagonal mixing in at least one basis state

    Note: rows with R_i = 0 (no off-diagonal elements) are excluded
    from the minimum, as their ratio is infinite and uninformative.
    """
    diag  = np.real(np.diag(H))
    radii = np.sum(np.abs(H), axis=1) - np.abs(diag)
    # exclude rows where R_i == 0 (purely diagonal rows, ratio = inf)
    mask  = radii > 0
    if not np.any(mask):
        return float("inf")
    ratio = np.abs(diag[mask]) / radii[mask]
    return float(np.min(ratio))


def compute_g3(H):
    """
    G3: Diagonal-dominance ratio at the HF-state proxy row.

    The row with the minimum diagonal element is used as a proxy for the
    Hartree-Fock (most-occupied) state — the configuration with the lowest
    single-determinant energy in a Z-diagonal Hamiltonian.

    G3 = |h_{i*,i*}| / R_{i*},   i* = argmin diag(H)

    Interpretation:
      G3 >> 1  : the HF-like state is strongly diagonally dominant
                 (off-diagonal perturbations are small; classical ansatz sufficient)
      G3 ~  1  : borderline; correlations moderate
      G3 << 1  : the HF state mixes strongly with other configurations
                 (strong correlation; single-reference methods insufficient)

    Scope / limitation:
      The PDF motivation discussed looking at the lowest few diagonal entries
      and their Gershgorin discs.  Here we use only the single row with the
      minimum diagonal element as a lightweight proxy.  This is useful for a
      compact summary, but it is not a full multireference diagnostic.

    WARNING: If R_{i*} ~ 0 (purely diagonal row), G3 = inf and is uninformative.
    This occurs for molecules where the minimum-diagonal basis state has no
    off-diagonal coupling (e.g. certain near-diagonal Hamiltonians).
    The returned value will be inf; treat with caution.
    """
    diag   = np.real(np.diag(H))
    radii  = np.sum(np.abs(H), axis=1) - np.abs(diag)
    i_star = int(np.argmin(diag))
    r      = radii[i_star]
    if r < 1e-14:
        return float("inf")
    return float(np.abs(diag[i_star]) / r)


def compute_g4(H):
    """
    G4: Ground-state Gershgorin disc isolation gap.

    Uses the minimum-diagonal row i* as a proxy for the ground state.
    Computes whether the Gershgorin disc of i* is isolated from all others:

      upper_bound(GS disc) = h_{i*,i*} + R_{i*}
      lower_bound(others)  = min_{i != i*}(h_{ii} - R_i)

      G4 = lower_bound(others) - upper_bound(GS)

    Interpretation:
      G4 > 0  : GS disc lies completely below all others with a finite gap
      G4 = 0  : GS disc touches another disc
      G4 < 0  : GS disc overlaps at least one other disc

    This definition follows the geometric intuition of separation more directly
    than the previous implementation by comparing the top of the candidate GS
    disc against the lowest point reached by any other disc.

    Scope / limitation:
      This follows the same single-row simplification as G3.  The PDF-level idea
      is closer to inspecting the lowest few candidate discs; here we only score
      the minimum-diagonal disc against the rest of the matrix.  It is therefore
      still a very partial proxy and should not be over-interpreted as a full
      single-reference / multi-reference criterion.
    """
    diag   = np.real(np.diag(H))
    radii  = np.sum(np.abs(H), axis=1) - np.abs(diag)
    i_star = int(np.argmin(diag))
    upper_gs    = diag[i_star] + radii[i_star]
    mask_others = np.ones(len(diag), dtype=bool)
    mask_others[i_star] = False
    lower_others = float(np.min(diag[mask_others] - radii[mask_others]))
    return float(lower_others - upper_gs)


# ── Per-molecule inspection ───────────────────────────────────────────────────

def inspect_molecule(mol_name, npz_path, charge=0):
    if not npz_path.exists():
        print(f"[SKIP] {npz_path.name}")
        return None

    data         = np.load(npz_path, allow_pickle=True)
    H            = data["hamiltonian"]
    weights      = data["weights"]
    eigvals      = data["eigvals"]
    energy_shift = float(data.get("energy_shift", 0.0))
    n_qubits     = int(np.log2(H.shape[0]))
    pauli_strings = data["pauli_strings"] if "pauli_strings" in data else None

    pm   = compute_pauli_metrics(weights, pauli_strings, n_qubits)
    g1   = compute_g1(H)
    g2   = compute_g2(H)
    g3   = compute_g3(H)
    g4   = compute_g4(H)
    gap  = eigvals[1] - eigvals[0]
    top5 = eigvals[:5] + energy_shift

    # ── Print ──────────────────────────────────────────────────────────────────
    miss       = "  [! no pauli_strings]" if pauli_strings is None else ""
    charge_tag = f"  charge={charge:+d}" if charge != 0 else ""
    print(f"\n{'='*60}")
    print(f"  {mol_name}  ({n_qubits} qubits){charge_tag}{miss}")
    print(f"{'='*60}")

    def f(v): return f"{v:.4f}" if not (isinstance(v, float) and np.isnan(v)) else "N/A"
    def fg(v): return f"{v:.4f}" if np.isfinite(v) else "  inf"

    print(f"  [1] Energy gap       Gap01        = {gap:.6f}")
    print(f"  [2] Qubit bias       Hub Score    = {f(pm['hub_score'])}"
          f"    Asymmetry = {f(pm['asymmetry'])}")
    if pauli_strings is not None:
        print(f"      Qubit importance             = {pm['q_importance']}")
    print(f"  [3] Pauli types      Z-only       = {f(pm['z_ratio'])}"
          f"    XY-only = {f(pm['xy_ratio'])}"
          f"    Mixed = {f(pm['mixed_ratio'])}")
    print(f"  [4] High-order ratio (>=4-body)   = {f(pm['high_order_ratio'])}")
    print(f"  [5] Top-5 full-space energies (with shift):")
    for i, e in enumerate(top5):
        print(f"        E{i} = {e:.8f}")
    print(f"  [G1] Diag-dom frac   G1           = {g1:.4f}"
          f"    (fraction of rows with |h_ii| >= R_i)")
    print(f"  [G2] Worst-case DD   G2           = {fg(g2)}"
          f"    (min |h_ii|/R_i over rows with R_i > 0)")
    print(f"  [G3] HF-state DD     G3           = {fg(g3)}"
          f"    (|h_ii|/R_i at min-diagonal row)")
    print(f"  [G4] GS disc sep     G4           = {g4:.4f}"
          f"    (lower(GS disc) - upper(other discs))")

    return dict(molecule=mol_name, charge=charge, qubits=n_qubits, gap01=gap,
                hub_score=pm["hub_score"], asymmetry=pm["asymmetry"],
                z_ratio=pm["z_ratio"], xy_ratio=pm["xy_ratio"],
                mixed_ratio=pm["mixed_ratio"],
                high_order=pm["high_order_ratio"], g1=g1, g2=g2, g3=g3, g4=g4)


# ── Molecule list — (name, filename, charge) ──────────────────────────────────
# charge=0: neutral; charge=1: singly charged cation
# Charged molecules have wrong-sector ground state in full-space diagonalization.

MOLECULES = [
    # L1: Minimalism
    ("H2_Equil",    "L1_H2_Equil_4q_geom_H_.0_.0_0.0;_H_.0_.0_0.735_jordan_wigner.npz",    0),
    ("BH",          "L1_BH_6q_geom_B_.0_.0_0.0;_H_.0_.0_1.232_jordan_wigner.npz",            0),
    # L2: Asymmetry
    ("BeH_Plus",    "L2_BeH_Plus_4q_geom_Be_.0_.0_0.0;_H_.0_.0_1.312_jordan_wigner.npz",    +1),
    ("LiH_Equil",   "L2_LiH_Equil_6q_geom_Li_.0_.0_0.0;_H_.0_.0_1.595_jordan_wigner.npz",   0),
    ("BF",          "L2_BF_8q_geom_B_.0_.0_0.0;_F_.0_.0_1.267_jordan_wigner.npz",            0),
    # L3: Stability
    ("HeH_Plus",    "L3_HeH_Plus_4q_geom_He_.0_.0_0.0;_H_.0_.0_0.774_jordan_wigner.npz",    +1),
    ("CH2_Singlet", "L3_CH2_Singlet_6q_geom_C_.0_.0_0.0;_H_.0_0.86_0.73;_H_.0_-0.86_0.73_jordan_wigner.npz", 0),
    ("LiH_Stretch", "L3_LiH_Stretch_6q_geom_Li_.0_.0_0.0;_H_.0_.0_3.500_jordan_wigner.npz",  0),
    ("H3_Triangle", "L3_H3_Triangle_6q_geom_H_.0_.0_0.0;_H_1.0_.0_0.0;_H_0.5_0.866_0.0_jordan_wigner.npz", +1),
    # L4: Representation
    ("H2_Stretch",  "L4_H2_Stretch_4q_geom_H_.0_.0_0.0;_H_.0_.0_2.5_jordan_wigner.npz",     0),
    ("H3_Linear",   "L4_H3_Linear_6q_geom_H_.0_.0_0.0;_H_.0_.0_1.0;_H_.0_.0_2.0_jordan_wigner.npz", +1),
    ("H2O",         "L4_H2O_StrongCorr_8q_geom_O_.0_.0_0.0;_H_.0_0.757_0.586;_H_.0_-0.757_0.586_jordan_wigner.npz", 0),
    # L5: Topology
    ("H4_Chain",    "L5_H4_Chain_8q_geom_H_.0_.0_0.0;_H_.0_.0_1.0;_H_.0_.0_2.0;_H_.0_.0_3.0_jordan_wigner.npz", 0),
    # L6: Scalability
    ("BeH2",        "L6_BeH2_Scalability_10q_geom_Be_.0_.0_0.0;_H_.0_.0_1.326;_H_.0_.0_-1.326_jordan_wigner.npz", 0),
]

# ── Main ──────────────────────────────────────────────────────────────────────

neutral_results = []
charged_results = []

print("\n" + "#"*70)
print("  NEUTRAL MOLECULES (charge = 0)  — FULL-SPACE fingerprints")
print("#"*70)
for name, fname, charge in MOLECULES:
    if charge == 0:
        result = inspect_molecule(name, MOL_DATA_DIR / fname, charge=charge)
        if result:
            neutral_results.append(result)

print("\n" + "#"*70)
print("  CHARGED MOLECULES (charge != 0) — FULL-SPACE fingerprints")
print("#"*70)
for name, fname, charge in MOLECULES:
    if charge != 0:
        result = inspect_molecule(name, MOL_DATA_DIR / fname, charge=charge)
        if result:
            charged_results.append(result)

# ── Helper for summary rows ────────────────────────────────────────────────────

def _fmt(v):
    if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
        return "   inf" if np.isinf(v) else "   N/A"
    return f"{v:6.3f}"

def _print_summary(results, title):
    print(f"\n{'='*85}")
    print(f"  {title}")
    print(f"{'='*85}")
    print(f"  {'Molecule':<14} {'q':>2}  {'Gap01':>8}  {'Z-ratio':>7}  "
          f"{'HighOrd':>7}  {'G1':>6}  {'G2':>6}  {'G3':>6}  {'G4':>7}")
    print("  " + "-"*83)
    for r in results:
        print(f"  {r['molecule']:<14} {r['qubits']:>2}  {r['gap01']:>8.4f}"
              f"  {_fmt(r['z_ratio'])}  {_fmt(r['high_order'])}"
              f"  {_fmt(r['g1'])}  {_fmt(r['g2'])}  {_fmt(r['g3'])}  {r['g4']:7.3f}")

_print_summary(neutral_results, "SUMMARY — NEUTRAL  (full-space / unrestricted)")
_print_summary(charged_results, "SUMMARY — CHARGED  (full-space, may be wrong sector)")

# ── G3 stability check for neutral molecules ───────────────────────────────────
print(f"\n{'='*85}")
print("  G3 STABILITY CHECK (neutral molecules only)")
print(f"{'='*85}")
print("  Rows with G3=inf indicate the min-diagonal basis state has R_i ~ 0")
print("  (purely diagonal row — no off-diagonal coupling — G3 uninformative)\n")
for r in neutral_results:
    status = "*** UNSTABLE (inf)" if np.isinf(r['g3']) else f"{r['g3']:.4f}"
    print(f"  {r['molecule']:<14}  G3 = {status}")
