import numpy as np
from collections import defaultdict
import pandas as pd
from pathlib import Path

np.set_printoptions(precision=4, suppress=True, linewidth=160)

MOL_DATA_DIR = Path(__file__).resolve().parent.parent / "mol_data"
OUT_DIR      = Path(__file__).resolve().parent


def _parse_wire_indices(op_str: str):
    wires = []
    i = 0
    while i < len(op_str):
        if op_str[i] == "(":
            j = i + 1
            while j < len(op_str) and op_str[j] != ")":
                j += 1
            wires.append(int(op_str[i + 1:j]))
            i = j
        i += 1
    return wires


def _pauli_type(op_str: str):
    if op_str in ("Identity", "I"):
        return "I"
    body = sum(ch in "XYZ" for ch in op_str)
    has_x, has_y, has_z = "X(" in op_str, "Y(" in op_str, "Z(" in op_str
    if has_z and not has_x and not has_y:   return f"{body}-body Z-only"
    if (has_x or has_y) and not has_z:      return f"{body}-body XY-only"
    if has_z and (has_x or has_y):          return f"{body}-body mixed"
    return f"{body}-body other"


def inspect_molecule(mol_name, npz_path):
    """Extract and print Hamiltonian fingerprint metrics for one molecule."""
    if not npz_path.exists():
        print(f"[SKIP] file not found: {npz_path}")
        return None

    data = np.load(npz_path, allow_pickle=True)
    weights       = data["weights"]
    pauli_strings = data["pauli_strings"]
    eigvals       = data["eigvals"]
    energy_shift  = float(data.get("energy_shift", 0.0))
    n_qubits      = int(np.log2(data["hamiltonian"].shape[0]))

    qubit_importance = np.zeros(n_qubits)
    total_weight = 0
    z_weight, xy_weight, mixed_weight = 0.0, 0.0, 0.0
    k_body_weight = defaultdict(float)

    for c, op in zip(weights, pauli_strings):
        w = abs(c)
        total_weight += w
        for q in _parse_wire_indices(str(op)):
            qubit_importance[q] += w
        ptype = _pauli_type(str(op))
        if ptype.endswith("Z-only"):   z_weight    += w
        elif "XY-only" in ptype:       xy_weight   += w
        elif "mixed"   in ptype:       mixed_weight += w
        body = sum(ch in "XYZ" for ch in str(op))
        k_body_weight[body] += w

    z_ratio          = z_weight    / total_weight
    xy_ratio         = xy_weight   / total_weight
    mixed_ratio      = mixed_weight / total_weight
    high_order_ratio = sum(v for k, v in k_body_weight.items() if k >= 4) / total_weight

    mean_imp  = np.mean(qubit_importance)
    hub_score = np.max(qubit_importance) / mean_imp   if mean_imp > 0 else 0.0
    asymmetry = (np.max(qubit_importance) - np.min(qubit_importance)) / mean_imp \
                if mean_imp > 0 else 0.0
    gap01 = eigvals[1] - eigvals[0]
    top5  = eigvals[:5] + energy_shift

    print(f"\n{'-'*60}")
    print(f"Molecule: {mol_name}  ({n_qubits} qubits)")
    print(f"  [1] Energy gap      : Gap01 = {gap01:8.6f}")
    print(f"  [2] Weight bias     : Hub Score = {hub_score:.4f}  Asymmetry = {asymmetry:.4f}")
    print(f"  [3] Qubit importance: {qubit_importance}")
    print(f"  [4] Operator types  : Z={z_ratio:.2%}  XY={xy_ratio:.2%}  Mixed={mixed_ratio:.2%}")
    print(f"  [5] High-order ratio: (>=4-body) = {high_order_ratio:.2%}")
    print(f"  [6] Top-5 energies (with shift):")
    for i, e in enumerate(top5):
        print(f"        E{i}: {e:.8f}")
    print(f"{'-'*60}")

    return {
        "molecule":    mol_name,
        "qubits":      n_qubits,
        "gap01":       gap01,
        "hub_score":   hub_score,
        "asymmetry":   asymmetry,
        "z_ratio":     z_ratio,
        "xy_ratio":    xy_ratio,
        "mixed_ratio": mixed_ratio,
        "high_order":  high_order_ratio,
        "q_imp_vector": qubit_importance.tolist(),
        "e0":          top5[0],
    }


# ── Molecules to inspect ──────────────────────────────────────────────────────

molecules = [
    ("H2_Equil",    "L1_H2_Equil_4q_geom_H_.0_.0_0.0;_H_.0_.0_0.735_jordan_wigner.npz"),
    ("BH",          "L1_BH_6q_geom_B_.0_.0_0.0;_H_.0_.0_1.232_jordan_wigner.npz"),
    ("BeH_Plus",    "L2_BeH_Plus_4q_geom_Be_.0_.0_0.0;_H_.0_.0_1.312_jordan_wigner.npz"),
    ("LiH_Equil",   "L2_LiH_Equil_6q_geom_Li_.0_.0_0.0;_H_.0_.0_1.595_jordan_wigner.npz"),
    ("BF",          "L2_BF_8q_geom_B_.0_.0_0.0;_F_.0_.0_1.267_jordan_wigner.npz"),
    ("HeH_Plus",    "L3_HeH_Plus_4q_geom_He_.0_.0_0.0;_H_.0_.0_0.774_jordan_wigner.npz"),
    ("CH2_Singlet", "L3_CH2_Singlet_6q_geom_C_.0_.0_0.0;_H_.0_0.86_0.73;_H_.0_-0.86_0.73_jordan_wigner.npz"),
    ("LiH_Stretch", "L3_LiH_Stretch_6q_geom_Li_.0_.0_0.0;_H_.0_.0_3.500_jordan_wigner.npz"),
    ("H3_Triangle", "L3_H3_Triangle_6q_geom_H_.0_.0_0.0;_H_1.0_.0_0.0;_H_0.5_0.866_0.0_jordan_wigner.npz"),
    ("H2_Stretch",  "L4_H2_Stretch_4q_geom_H_.0_.0_0.0;_H_.0_.0_2.5_jordan_wigner.npz"),
    ("H3_Linear",   "L4_H3_Linear_6q_geom_H_.0_.0_0.0;_H_.0_.0_1.0;_H_.0_.0_2.0_jordan_wigner.npz"),
    ("H2O",         "L4_H2O_StrongCorr_8q_geom_O_.0_.0_0.0;_H_.0_0.757_0.586;_H_.0_-0.757_0.586_jordan_wigner.npz"),
    ("H4_Chain",    "L5_H4_Chain_8q_geom_H_.0_.0_0.0;_H_.0_.0_1.0;_H_.0_.0_2.0;_H_.0_.0_3.0_jordan_wigner.npz"),
    ("BeH2",        "L6_BeH2_Scalability_10q_geom_Be_.0_.0_0.0;_H_.0_.0_1.326;_H_.0_.0_-1.326_jordan_wigner.npz"),
]

all_data = []
for name, fname in molecules:
    res = inspect_molecule(name, MOL_DATA_DIR / fname)
    if res:
        all_data.append(res)

df = pd.DataFrame(all_data)
summary_cols = ["molecule", "qubits", "gap01", "hub_score", "z_ratio", "high_order"]
print("\n### Summary Table (sorted by Hub Score) ###")
print(df.sort_values("hub_score", ascending=False)[summary_cols])

csv_path = OUT_DIR / "molecular_fingerprints_full.csv"
df.to_csv(csv_path, index=False)
print(f"\n[OK] CSV saved -> {csv_path}")
