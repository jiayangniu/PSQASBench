"""
compare_sector_vs_full.py

对每个分子并排计算 G1-G4 两次：
  - Full  : 在完整 2^n × 2^n 哈密顿量矩阵上
  - Sector: 在物理粒子数子空间 H_sector 上（利用 mol_data_physical 中存储的 sector_indices）

读取来源: PSQASBench/mol_data_physical/  （包含 hamiltonian + sector_indices）

说明：
  这里沿用 read_fingerprints.py 中当前版本的 G1-G4 定义。
  其中 G1/G2 是全局对角占优摘要；G3/G4 是基于“最小对角元所在行”的
  单候选 Gershgorin 启发式指标。PDF 中更接近“看最低几个对角元 / 圆盘”，
  但这里暂时保留单候选实现，只在注释中明确其局限性。
"""

import numpy as np
from pathlib import Path

PHYS_DIR = Path(__file__).resolve().parent.parent / "mol_data_physical"

# ── Gershgorin 指标函数（对任意方阵均适用）────────────────────────────────────

def g1(H):
    """对角占优行的比例。"""
    diag  = np.real(np.diag(H))
    radii = np.sum(np.abs(H), axis=1) - np.abs(diag)
    return float(np.sum(np.abs(diag) >= radii)) / H.shape[0]

def g2(H):
    """最坏情况对角占优比率 min |h_ii|/R_i （R_i>0 的行）。"""
    diag  = np.real(np.diag(H))
    radii = np.sum(np.abs(H), axis=1) - np.abs(diag)
    mask  = radii > 1e-14
    if not np.any(mask):
        return float("inf")
    return float(np.min(np.abs(diag[mask]) / radii[mask]))

def g3(H):
    """最小对角元所在行的 |h_ii|/R_i（HF 态代理）。

    注意：这是单候选近似。更贴近 PDF 原意的做法会考察最低几个对角元
    对应的 Gershgorin 圆盘，但这里为了保持当前分析一致性，只取一个代理行。
    """
    diag   = np.real(np.diag(H))
    radii  = np.sum(np.abs(H), axis=1) - np.abs(diag)
    i_star = int(np.argmin(diag))
    r = radii[i_star]
    if r < 1e-14:
        return float("inf")
    return float(np.abs(diag[i_star]) / r)

def g4(H):
    """基态圆盘分离度：min lower(other discs) − upper(GS disc)。

    注意：这里比较的是“最小对角元对应圆盘”的上边界与其他圆盘中最低下边界，
    因而更符合“最低圆盘是否与其余圆盘分离”的直觉。
    但它仍然只使用一个代理圆盘，而不是 PDF 设想中的“最低几个圆盘”的
    整体重叠 / 分离分析，所以仍然是片面的单候选 heuristic。
    """
    diag   = np.real(np.diag(H))
    radii  = np.sum(np.abs(H), axis=1) - np.abs(diag)
    i_star = int(np.argmin(diag))
    upper_gs = diag[i_star] + radii[i_star]
    mask = np.ones(len(diag), dtype=bool)
    mask[i_star] = False
    lower_others = float(np.min(diag[mask] - radii[mask]))
    return float(lower_others - upper_gs)

def fmt(v, w=7):
    if np.isinf(v):  return f"{'inf':>{w}}"
    if np.isnan(v):  return f"{'N/A':>{w}}"
    return f"{v:{w}.4f}"

# ── 分子列表：(显示名, npz文件名, charge) ─────────────────────────────────────

MOLECULES = [
    # 中性
    ("H2_Equil",    "L1_H2_Equil_4q_geom_H_.0_.0_0.0;_H_.0_.0_0.735_jordan_wigner.npz",    0),
    ("BH",          "L1_BH_6q_geom_B_.0_.0_0.0;_H_.0_.0_1.232_jordan_wigner.npz",            0),
    ("LiH_Equil",   "L2_LiH_Equil_6q_geom_Li_.0_.0_0.0;_H_.0_.0_1.595_jordan_wigner.npz",   0),
    ("BF",          "L2_BF_8q_geom_B_.0_.0_0.0;_F_.0_.0_1.267_jordan_wigner.npz",            0),
    ("CH2_Singlet", "L3_CH2_Singlet_6q_geom_C_.0_.0_0.0;_H_.0_0.86_0.73;_H_.0_-0.86_0.73_jordan_wigner.npz", 0),
    ("LiH_Stretch", "L3_LiH_Stretch_6q_geom_Li_.0_.0_0.0;_H_.0_.0_3.500_jordan_wigner.npz",  0),
    ("H2_Stretch",  "L4_H2_Stretch_4q_geom_H_.0_.0_0.0;_H_.0_.0_2.5_jordan_wigner.npz",     0),
    ("H2O",         "L4_H2O_StrongCorr_8q_geom_O_.0_.0_0.0;_H_.0_1.186_0.918;_H_.0_-1.186_0.918_jordan_wigner.npz", 0),
    ("H4_Chain",    "L5_H4_Chain_8q_geom_H_.0_.0_0.0;_H_.0_.0_1.0;_H_.0_.0_2.0;_H_.0_.0_3.0_jordan_wigner.npz", 0),
    ("BeH2",        "L6_BeH2_Scalability_10q_geom_Be_.0_.0_0.0;_H_.0_.0_1.326;_H_.0_.0_-1.326_jordan_wigner.npz", 0),
    # 带电
    ("BeH_Plus",    "L2_BeH_Plus_4q_geom_Be_.0_.0_0.0;_H_.0_.0_1.312_jordan_wigner.npz",    +1),
    ("HeH_Plus",    "L3_HeH_Plus_4q_geom_He_.0_.0_0.0;_H_.0_.0_0.774_jordan_wigner.npz",    +1),
    ("H3_Triangle", "L3_H3_Triangle_6q_geom_H_.0_.0_0.0;_H_1.0_.0_0.0;_H_0.5_0.866_0.0_jordan_wigner.npz", +1),
    ("H3_Linear",   "L4_H3_Linear_6q_geom_H_.0_.0_0.0;_H_.0_.0_1.0;_H_.0_.0_2.0_jordan_wigner.npz", +1),
]

# ── 主循环 ────────────────────────────────────────────────────────────────────

rows = []   # 用于最后的汇总表

for mol_name, fname, charge in MOLECULES:
    path = PHYS_DIR / fname
    if not path.exists():
        print(f"[SKIP] {fname}")
        continue

    data          = np.load(path, allow_pickle=True)
    H_full        = data["hamiltonian"]                     # 完整 2^n × 2^n
    sector_idx    = data["sector_indices"].astype(int)      # 物理子空间行列索引
    n_alpha       = int(data["n_alpha"])
    n_beta        = int(data["n_beta"])
    n_qubits      = int(np.log2(H_full.shape[0]))
    sector_size   = len(sector_idx)

    # 提取物理子空间矩阵（块对角中的正确块）
    H_sector = H_full[np.ix_(sector_idx, sector_idx)]
    H_sector = np.real(H_sector)   # 物理子空间哈密顿量是实矩阵

    # ── 计算两套指标 ─────────────────────────────────────────────────────────
    results = {}
    for tag, H in [("full", H_full), ("sector", H_sector)]:
        results[tag] = dict(g1=g1(H), g2=g2(H), g3=g3(H), g4=g4(H), dim=H.shape[0])

    # ── 逐分子打印 ───────────────────────────────────────────────────────────
    charge_tag = f" charge={charge:+d}" if charge != 0 else " neutral"
    print(f"\n{'='*66}")
    print(f"  {mol_name}  ({n_qubits}q,{charge_tag})  "
          f"sector=({n_alpha}α+{n_beta}β)  "
          f"dim: {H_full.shape[0]} → {sector_size}")
    print(f"{'='*66}")
    print(f"  {'':10}  {'G1':>7}  {'G2':>7}  {'G3':>7}  {'G4':>8}")
    print(f"  {'Full H':10}  "
          f"{fmt(results['full']['g1'])}  "
          f"{fmt(results['full']['g2'])}  "
          f"{fmt(results['full']['g3'])}  "
          f"{fmt(results['full']['g4'], 8)}")
    print(f"  {'H_sector':10}  "
          f"{fmt(results['sector']['g1'])}  "
          f"{fmt(results['sector']['g2'])}  "
          f"{fmt(results['sector']['g3'])}  "
          f"{fmt(results['sector']['g4'], 8)}")

    # 差值（inf 跳过）
    diffs = {}
    for k in ("g1", "g2", "g3", "g4"):
        vf, vs = results["full"][k], results["sector"][k]
        diffs[k] = (vs - vf) if (np.isfinite(vf) and np.isfinite(vs)) else float("nan")
    print(f"  {'Δ(sec-full)':10}  "
          f"{fmt(diffs['g1'])}  "
          f"{fmt(diffs['g2'])}  "
          f"{fmt(diffs['g3'])}  "
          f"{fmt(diffs['g4'], 8)}")

    rows.append(dict(name=mol_name, charge=charge, q=n_qubits,
                     full_dim=H_full.shape[0], sec_dim=sector_size,
                     **{f"{k}_full":   results["full"][k]   for k in ("g1","g2","g3","g4")},
                     **{f"{k}_sector": results["sector"][k] for k in ("g1","g2","g3","g4")},
                     **{f"d{k}":       diffs[k]             for k in ("g1","g2","g3","g4")}))

# ── 汇总对比表 ────────────────────────────────────────────────────────────────

def section(title, subset):
    print(f"\n{'='*95}")
    print(f"  {title}")
    print(f"{'='*95}")
    hdr = (f"  {'Molecule':<14} {'q':>2} {'dim→sec':>10}  "
           f"{'G1_full':>7} {'G1_sec':>7} {'ΔG1':>7}  "
           f"{'G2_full':>7} {'G2_sec':>7} {'ΔG2':>7}  "
           f"{'G3_full':>7} {'G3_sec':>7}  "
           f"{'G4_full':>8} {'G4_sec':>8}")
    print(hdr)
    print("  " + "-"*93)
    for r in subset:
        dim_str = f"{r['full_dim']}→{r['sec_dim']}"
        print(f"  {r['name']:<14} {r['q']:>2} {dim_str:>10}  "
              f"{fmt(r['g1_full'])} {fmt(r['g1_sector'])} {fmt(r['dg1'])}  "
              f"{fmt(r['g2_full'])} {fmt(r['g2_sector'])} {fmt(r['dg2'])}  "
              f"{fmt(r['g3_full'])} {fmt(r['g3_sector'])}  "
              f"{fmt(r['g4_full'],8)} {fmt(r['g4_sector'],8)}")

neutral = [r for r in rows if r["charge"] == 0]
charged = [r for r in rows if r["charge"] != 0]

section("NEUTRAL MOLECULES — Full H  vs  H_sector", neutral)
section("CHARGED MOLECULES — Full H  vs  H_sector", charged)

# ── G3 变化汇总（重点关注 inf→有限 的转变）────────────────────────────────────
print(f"\n{'='*60}")
print("  G3 变化汇总（inf 表示对应行 R_i ≈ 0）")
print(f"{'='*60}")
for r in rows:
    gf = "inf" if np.isinf(r["g3_full"])   else f"{r['g3_full']:.4f}"
    gs = "inf" if np.isinf(r["g3_sector"]) else f"{r['g3_sector']:.4f}"
    tag = ""
    if np.isinf(r["g3_full"]) and np.isfinite(r["g3_sector"]):
        tag = "  ← inf 消失！子空间修复"
    elif np.isfinite(r["g3_full"]) and np.isinf(r["g3_sector"]):
        tag = "  ← 子空间引入 inf"
    charge_str = f"charge={r['charge']:+d}" if r["charge"] != 0 else "neutral"
    print(f"  {r['name']:<14} ({charge_str:>10})  Full={gf:>9}  Sector={gs:>9}{tag}")
