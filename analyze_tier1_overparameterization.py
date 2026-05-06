"""
Tier 1 over-parameterization analysis.

For each method × molecule × depth × seed, parse episode_traces.txt and
collect gate counts (rot, cnot) from all analysis_snapshots that are within
chemical accuracy.  Report mean ± std across all qualifying circuits per run,
then aggregate across seeds.

Usage:
    python analyze_tier1_overparameterization.py
"""

import ast
import os
import re
import json
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).parent / "results"

# Chemical accuracy threshold (Ha)
CHEM_ACC = 0.0016

# Target molecules and their expected best-error fingerprints (for reference)
TARGETS = {
    "L1_BeH2_STO3G_6q": 5.54e-4,
    "L2_LiH_Equil_6q":  1.05e-3,
}

SEEDS = ["seed22222", "seed33333"]

# ── run directories to analyse ─────────────────────────────────────────────
# Discover all relevant run dirs automatically
def discover_runs():
    runs = []
    for method_dir in ROOT.iterdir():
        if not method_dir.is_dir():
            continue
        method = method_dir.name
        for mol, _ in TARGETS.items():
            mol_dir = method_dir / mol
            if not mol_dir.exists():
                continue
            for exp_dir in mol_dir.rglob("episode_traces.txt"):
                seed_dir = exp_dir.parent
                seed = seed_dir.name
                if seed not in SEEDS:
                    continue
                config_path = seed_dir.name  # just seed name
                # infer depth from directory name
                parts = str(seed_dir.relative_to(mol_dir)).split(os.sep)
                depth_tag = "shallow" if "depth10" in str(seed_dir) else "deep" if "depth50" in str(seed_dir) else "single"
                runs.append({
                    "method": method,
                    "mol": mol,
                    "depth": depth_tag,
                    "seed": seed,
                    "trace_path": exp_dir,
                })
    return runs


# ── trace parsers ──────────────────────────────────────────────────────────

def parse_episodes(text):
    """Split trace file into episode blocks, return list of dicts."""
    episodes = []
    blocks = re.split(r'\[episode \d+\]', text)
    for block in blocks[1:]:  # skip preamble
        ep = {}
        # energy_errors_ha
        m = re.search(r'energy_errors_ha\s*=\s*(\[.*?\])', block, re.S)
        if m:
            try:
                ep['energy_errors_ha'] = ast.literal_eval(m.group(1))
            except Exception:
                ep['energy_errors_ha'] = []
        # analysis_snapshots
        m = re.search(r'analysis_snapshots\s*=\s*(\[.*?\])\s*(?:\n|$)', block, re.S)
        if m:
            raw = m.group(1)
            try:
                ep['analysis_snapshots'] = ast.literal_eval(raw)
            except Exception:
                try:
                    ep['analysis_snapshots'] = json.loads(raw)
                except Exception:
                    ep['analysis_snapshots'] = []
        else:
            ep['analysis_snapshots'] = []
        # actions (for HyRLQAS act_param check)
        m = re.search(r'^actions\s*=\s*(\[.*?\])\s*$', block, re.M | re.S)
        if m:
            try:
                ep['actions'] = ast.literal_eval(m.group(1))
            except Exception:
                ep['actions'] = None
        else:
            ep['actions'] = None
        episodes.append(ep)
    return episodes


def gate_counts_from_snapshot(snap, actions=None, energy_errors=None):
    """
    Return (rot_count, cnot_count, energy_error) for a single snapshot.
    Returns None if the circuit is not within chemical accuracy.
    """
    # ── gates_direct format (GQEQAS, qdarts, TFQAS) ────────────────────
    if 'gates_direct' in snap:
        gates = snap['gates_direct']
        rot  = sum(1 for g in gates if g.get('type') == 'rot')
        cnot = sum(1 for g in gates if g.get('type') == 'cnot')
        # energy error: from energy_errors_ha of the episode (single value)
        err = None
        if energy_errors and len(energy_errors) > 0:
            err = energy_errors[-1]  # GQEQAS stores final circuit energy
        return rot, cnot, err

    # ── param_step_indices format (CRLQAS, HyRLQAS) ────────────────────
    if 'param_step_indices' in snap:
        step = snap['step']
        psi  = snap['param_step_indices']
        total_gates = step + 1
        rot = len(psi)
        cnot = total_gates - rot

        err = None
        if energy_errors and step < len(energy_errors):
            err = energy_errors[step]
        return rot, cnot, err

    return None


def analyse_run(trace_path):
    """Return list of (rot, cnot, energy_error_ha) for all chem-acc snapshots."""
    text = trace_path.read_text(errors='replace')
    episodes = parse_episodes(text)
    results = []
    for ep in episodes:
        for snap in ep.get('analysis_snapshots', []):
            gc = gate_counts_from_snapshot(
                snap,
                actions=ep.get('actions'),
                energy_errors=ep.get('energy_errors_ha', []),
            )
            if gc is None:
                continue
            rot, cnot, err = gc
            if err is None:
                # accept if no energy info (qdarts non-feasible episodes filtered later)
                continue
            if err < CHEM_ACC:
                results.append((rot, cnot, err))
    return results


# ── main ───────────────────────────────────────────────────────────────────

def mean_std(vals):
    if not vals:
        return float('nan'), float('nan'), 0
    if len(vals) == 1:
        return vals[0], float('nan'), 1
    return statistics.mean(vals), statistics.stdev(vals), len(vals)


def main():
    runs = discover_runs()

    # group by (method, mol, depth)
    grouped = defaultdict(list)
    for r in runs:
        key = (r['method'], r['mol'], r['depth'])
        grouped[key].append(r)

    # print header
    print(f"{'Method':<16} {'Molecule':<22} {'Depth':<8} "
          f"{'n_circuits':>10} {'mean_rot':>9} {'std_rot':>8} "
          f"{'mean_cnot':>10} {'std_cnot':>9} {'mean_total':>11}")
    print("-" * 105)

    # sort for readability
    order = ['crlqas', 'hyrlqas', 'qdarts', 'gqeqas', 'tfqas']
    all_keys = sorted(grouped.keys(), key=lambda k: (TARGETS.get(k[1], 0), order.index(k[0]) if k[0] in order else 99, k[2]))

    # also build a compact table for the paper
    paper_rows = []

    for key in all_keys:
        run_list = grouped[key]
        all_rot, all_cnot, all_total = [], [], []
        for r in run_list:
            for rot, cnot, err in analyse_run(r['trace_path']):
                all_rot.append(rot)
                all_cnot.append(cnot)
                all_total.append(rot + cnot)

        mr, sr, n = mean_std(all_rot)
        mc, sc, _ = mean_std(all_cnot)
        mt, st, _ = mean_std(all_total)

        mol_short = key[1].replace("L1_", "T1_").replace("L2_", "T1_").replace("_STO3G_6q","").replace("_6q","")
        print(f"{key[0]:<16} {key[1]:<22} {key[2]:<8} "
              f"{n:>10} {mr:>9.2f} {sr:>8.2f} "
              f"{mc:>10.2f} {sc:>9.2f} {mt:>11.2f}")

        paper_rows.append({
            'method': key[0],
            'mol': key[1],
            'depth': key[2],
            'n': n,
            'mean_rot': mr, 'std_rot': sr,
            'mean_cnot': mc, 'std_cnot': sc,
            'mean_total': mt,
        })

    # ── compact paper table ────────────────────────────────────────────
    print("\n\n=== PAPER TABLE: mean CNOT and total gate count for Tier-1 circuits ===")
    print("(all circuits within chemical accuracy, across both seeds)")
    print()

    METHOD_LABEL = {'crlqas': 'CRLQAS', 'hyrlqas': 'HyRLQAS',
                    'qdarts': 'QuantumDARTS', 'gqeqas': 'GQEQAS', 'tfqas': 'TFQAS'}
    DEPTH_LABEL  = {'shallow': 'Shallow (≤10)', 'deep': 'Deep (≤50)', 'single': 'Single'}

    for mol_key in TARGETS:
        mol_short = "BeH2_STO3G" if "BeH2" in mol_key else "LiH_Equil"
        print(f"--- {mol_short} ---")
        header = f"{'Method':<14}"
        for d in ['shallow', 'deep', 'single']:
            if any(r['mol'] == mol_key and r['depth'] == d for r in paper_rows):
                header += f"  {'CNOT ('+d+')':>14}  {'Total ('+d+')':>14}"
        print(header)

        for m in order:
            row = f"{METHOD_LABEL.get(m, m):<14}"
            for d in ['shallow', 'deep', 'single']:
                match = [r for r in paper_rows if r['method'] == m and r['mol'] == mol_key and r['depth'] == d]
                if match:
                    r = match[0]
                    cnot_s = f"{r['mean_cnot']:.1f}±{r['std_cnot']:.1f}" if r['n'] > 0 else "—"
                    tot_s  = f"{r['mean_total']:.1f}±{r['std_total'] if 'std_total' in r else 0:.1f}"
                    # recompute std_total
                    row += f"  {cnot_s:>14}  {tot_s:>14}"
            print(row)
        print()

    # ── redundancy computation ─────────────────────────────────────────
    # minimal circuit from critical-structure analysis:
    # BeH2: 0 CNOT, 2-3 rot (let's use 2 as the theoretical minimum)
    # LiH:  0 CNOT, 2-3 rot
    MINIMAL = {
        "L1_BeH2_STO3G_6q": {"cnot": 0, "rot": 2},
        "L2_LiH_Equil_6q":  {"cnot": 0, "rot": 2},
    }

    print("\n=== REDUNDANCY = mean_total - minimal_total ===")
    print(f"{'Method':<14} {'Molecule':<22} {'Depth':<8} "
          f"{'mean_total':>11} {'minimal':>8} {'redundancy':>11} {'red_%':>8}")
    print("-" * 90)
    for r in sorted(paper_rows, key=lambda x: (x['mol'], order.index(x['method']) if x['method'] in order else 99, x['depth'])):
        minimal = MINIMAL.get(r['mol'], {})
        min_total = minimal.get('cnot', 0) + minimal.get('rot', 2)
        redundancy = r['mean_total'] - min_total if r['n'] > 0 else float('nan')
        red_pct = 100 * redundancy / r['mean_total'] if r['mean_total'] > 0 else float('nan')
        print(f"{METHOD_LABEL.get(r['method'], r['method']):<14} {r['mol']:<22} {r['depth']:<8} "
              f"{r['mean_total']:>11.2f} {min_total:>8} {redundancy:>11.2f} {red_pct:>7.1f}%")


if __name__ == '__main__':
    main()
