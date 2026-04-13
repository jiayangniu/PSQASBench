"""Compute fingerprint metrics for the BeH2 basis-series dataset.

The script prefers files under `mol_data_beh2_basis/`.  If that directory is
empty, it falls back to the BeH2 files stored in `mol_data/`, which is useful
while the dataset is being renamed or reorganized.

For large sparse runs the generator may omit the dense Hamiltonian matrix; in
that case the script still reports spectrum and Pauli-decomposition summaries,
and prints `N/A` for the Gershgorin-style matrix metrics.
"""

import numpy as np
from collections import defaultdict
from pathlib import Path

np.set_printoptions(precision=4, suppress=True, linewidth=160)

PRIMARY_DATA_DIR = Path(__file__).resolve().parent.parent / "mol_data_beh2_basis"
FALLBACK_DATA_DIR = Path(__file__).resolve().parent.parent / "mol_data"
FILE_PATTERNS = ("L1_BeH2_*.npz", "L6_BeH2_*.npz")


def _discover_files():
    """Return BeH2 basis-series files, preferring the dedicated data directory."""
    for data_dir in (PRIMARY_DATA_DIR, FALLBACK_DATA_DIR):
        files = []
        seen = set()
        if not data_dir.exists():
            continue
        for pattern in FILE_PATTERNS:
            for path in sorted(data_dir.glob(pattern)):
                if path.name not in seen:
                    files.append(path)
                    seen.add(path.name)
        if files:
            return files, data_dir
    return [], PRIMARY_DATA_DIR


def _wire_indices(op_str: str):
    """Return qubit indices appearing in a PennyLane-style Pauli string."""
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
    """Compute fingerprint metrics derived from the Pauli decomposition."""
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
    """Fraction of rows satisfying the Gershgorin diagonal-dominance test."""
    diag = np.real(np.diag(H))
    radii = np.sum(np.abs(H), axis=1) - np.abs(diag)
    return float(np.sum(np.abs(diag) >= radii)) / H.shape[0]


def compute_g2(H):
    """Worst-case diagonal-dominance ratio across all rows."""
    diag = np.real(np.diag(H))
    radii = np.sum(np.abs(H), axis=1) - np.abs(diag)
    mask = radii > 0
    if not np.any(mask):
        return float("inf")
    return float(np.min(np.abs(diag[mask]) / radii[mask]))


def compute_g3(H):
    """Diagonal-dominance ratio on the minimum-diagonal row."""
    diag = np.real(np.diag(H))
    radii = np.sum(np.abs(H), axis=1) - np.abs(diag)
    i_star = int(np.argmin(diag))
    r = radii[i_star]
    if r < 1e-14:
        return float("inf")
    return float(np.abs(diag[i_star]) / r)


def compute_g4(H):
    """Separation between the candidate ground-state disc and the rest."""
    diag = np.real(np.diag(H))
    radii = np.sum(np.abs(H), axis=1) - np.abs(diag)
    i_star = int(np.argmin(diag))
    upper_gs = diag[i_star] + radii[i_star]
    mask_others = np.ones(len(diag), dtype=bool)
    mask_others[i_star] = False
    lower_others = float(np.min(diag[mask_others] - radii[mask_others]))
    return float(lower_others - upper_gs)


def inspect_file(npz_path: Path):
    data = np.load(npz_path, allow_pickle=True)
    H = data["hamiltonian"] if "hamiltonian" in data else None
    eigvals = data["eigvals"]
    weights = data["weights"]
    pauli_strings = data["pauli_strings"]
    n_qubits = int(data["n_qubits"]) if "n_qubits" in data else int(np.log2(H.shape[0]))
    basis = str(data["basis"]) if "basis" in data else "unknown"
    active_electrons = int(data["active_electrons"]) if "active_electrons" in data else -1
    active_orbitals = int(data["active_orbitals"]) if "active_orbitals" in data else -1
    energy_shift = float(data.get("energy_shift", 0.0))

    pm = compute_pauli_metrics(weights, pauli_strings, n_qubits)
    if H is not None:
        g1 = compute_g1(H)
        g2 = compute_g2(H)
        g3 = compute_g3(H)
        g4 = compute_g4(H)
    else:
        g1 = g2 = g3 = g4 = float("nan")
    gap = float(eigvals[1] - eigvals[0]) if len(eigvals) > 1 else float("nan")
    top5 = eigvals[:5] + energy_shift

    def _metric_text(value):
        if np.isnan(value):
            return "N/A"
        if np.isinf(value):
            return "inf"
        return f"{value:.4f}"

    print(f"\n{'=' * 70}")
    print(f"  {npz_path.stem}")
    print(f"{'=' * 70}")
    print(f"  basis              = {basis}")
    print(f"  active electrons   = {active_electrons}")
    print(f"  active orbitals    = {active_orbitals}")
    print(f"  qubits             = {n_qubits}")
    print(f"  Gap01              = {gap:.6f}")
    print(f"  Hub Score          = {pm['hub_score']:.4f}")
    print(f"  Asymmetry          = {pm['asymmetry']:.4f}")
    print(f"  Z-only             = {pm['z_ratio']:.4f}")
    print(f"  XY-only            = {pm['xy_ratio']:.4f}")
    print(f"  Mixed              = {pm['mixed_ratio']:.4f}")
    print(f"  High-order ratio   = {pm['high_order_ratio']:.4f}")
    print(f"  G1                 = {_metric_text(g1)}")
    print(f"  G2                 = {_metric_text(g2)}")
    print(f"  G3                 = {_metric_text(g3)}")
    print(f"  G4                 = {_metric_text(g4)}")
    print(f"  Top-5 energies     = {top5}")
    if H is None:
        print("  Note               = dense Hamiltonian not stored; G1-G4 unavailable")

    return dict(
        molecule=npz_path.stem,
        basis=basis,
        active_orbitals=active_orbitals,
        qubits=n_qubits,
        gap01=gap,
        asymmetry=pm["asymmetry"],
        z_ratio=pm["z_ratio"],
        high_order=pm["high_order_ratio"],
        g1=g1,
        g2=g2,
        g3=g3,
        g4=g4,
    )


def _fmt(v):
    if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
        return "   inf" if np.isinf(v) else "   N/A"
    return f"{v:6.3f}"


def main():
    files, data_dir = _discover_files()
    if not files:
        print(
            "No BeH2 basis-series .npz files found in "
            f"{PRIMARY_DATA_DIR} or {FALLBACK_DATA_DIR}"
        )
        return

    results = [inspect_file(path) for path in files]
    results.sort(key=lambda r: (r["qubits"], r["basis"], r["molecule"]))

    print(f"Reading files from: {data_dir}")

    print(f"\n{'=' * 96}")
    print("  SUMMARY — BeH2 basis series")
    print(f"{'=' * 96}")
    print(
        f"  {'Molecule':<28} {'Basis':<10} {'q':>2}  {'Gap01':>8}  "
        f"{'Asym':>6}  {'Z-ratio':>7}  {'HighOrd':>7}  {'G1':>6}  {'G2':>6}  {'G3':>6}  {'G4':>7}"
    )
    print("  " + "-" * 94)
    for r in results:
        print(
            f"  {r['molecule']:<28} {r['basis']:<10} {r['qubits']:>2}  {r['gap01']:>8.4f}  "
            f"{_fmt(r['asymmetry'])}  {_fmt(r['z_ratio'])}  {_fmt(r['high_order'])}  "
            f"{_fmt(r['g1'])}  {_fmt(r['g2'])}  {_fmt(r['g3'])}  {_fmt(r['g4'])}"
        )


if __name__ == "__main__":
    main()
