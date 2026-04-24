#!/usr/bin/env python3
"""
analyze_fidelity.py — Pairwise quantum-state fidelity analysis for pruned circuits.

For each critical_structure_analysis/<bucket_dir>/ that contains a summary.tsv,
this script:
  1. Parses all retained circuit structures from the 'retained_gates' column.
  2. Re-optimizes each circuit independently using the chosen optimizer.
  3. Computes pairwise fidelities |⟨ψ_i|ψ_j⟩|².
  4. Clusters circuits with deterministic complete-linkage agglomerative
     clustering: every pair inside a cluster must individually satisfy the
     threshold, and merges are ordered by the strongest minimum cross-cluster
     fidelity.
  5. Writes fidelity_analysis.md and fidelity_matrix.tsv to the same directory.

Usage:
    python analyze_fidelity.py <analysis_dir> [<analysis_dir2> ...]
    python analyze_fidelity.py critical_structure_analysis/l1_beh2_depth10_bucket027/
    python analyze_fidelity.py critical_structure_analysis/  # process all subdirs

    # Use the same optimizer AND hyperparams as the training run (recommended):
    python analyze_fidelity.py <dir> --optimizer inherit
    # Force rotosolve explicitly:
    python analyze_fidelity.py <dir> --optimizer rotosolve --rotosolve-sweeps 4

The script uses the 'run_dir' field in summary.tsv to automatically locate the
.npz Hamiltonian file and training hyperparameters via config_used.cfg.

NOTE on what this tool measures:
  The fidelity comparison is between states produced by re-optimizing each
  retained circuit structure independently from scratch.  A result of
  n_clusters==1 means "all retained structures fall into one high-fidelity
  re-optimized state cluster at the chosen threshold".  It does NOT directly
  prove that the original retained structures belong to the same variational
  basin, nor does it prove exact physical identity beyond the chosen fidelity
  threshold.  It answers the question "given free choice of angles, do these
  CNOT skeletons converge to the same state family?"
"""
from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import numpy as np
from qulacs import QuantumState

from critical_structure_tool.io_utils import build_run_context
from critical_structure_tool.circuit_utils import (
    build_qulacs_circuit,
    optimize_circuit as _optimize_circuit,
)
from critical_structure_tool.types import GateSpec, RunContext

REPO_ROOT = Path(__file__).resolve().parent


# ── Argument parsing ───────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Pairwise quantum-state fidelity analysis for pruned circuits from "
            "critical_structure_tool output.  Reads summary.tsv, writes "
            "fidelity_analysis.md and fidelity_matrix.tsv to the same directory."
        )
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        help=(
            "One or more critical_structure_analysis/<bucket_dir>/ paths.  "
            "Each must contain a summary.tsv produced by critical_structure_tool.  "
            "If a parent directory is given, all subdirectories with summary.tsv "
            "are processed recursively."
        ),
    )
    parser.add_argument(
        "--optimizer",
        choices=["cobyla", "rotosolve", "inherit"],
        default="inherit",
        help=(
            "Optimizer for angle re-optimization.  "
            "'inherit' reads resolved_optimizer from summary.tsv AND inherits "
            "cobyla_maxiter / rotosolve_sweeps from the run's config_used.cfg "
            "(--cobyla-maxiter / --rotosolve-sweeps act as fallback defaults only). "
            "'cobyla' or 'rotosolve' override for all circuits and always use the "
            "CLI hyperparameters.  Default: inherit."
        ),
    )
    parser.add_argument(
        "--n-restarts",
        type=int,
        default=20,
        help=(
            "Number of independent restarts per circuit.  For COBYLA this is "
            "passed directly to circuit_utils.  For Rotosolve this controls how "
            "many random angle initializations are tried before sweeps.  Default: 20."
        ),
    )
    parser.add_argument(
        "--cobyla-maxiter",
        type=int,
        default=2000,
        help=(
            "COBYLA max iterations per restart.  Used as fallback when "
            "--optimizer inherit cannot read the value from config_used.cfg.  "
            "Default: 2000."
        ),
    )
    parser.add_argument(
        "--rotosolve-sweeps",
        type=int,
        default=2,
        help=(
            "Number of Rotosolve sweeps per restart.  Used as fallback when "
            "--optimizer inherit cannot read the value from config_used.cfg.  "
            "Default: 2."
        ),
    )
    parser.add_argument(
        "--fidelity-threshold",
        type=float,
        default=0.999,
        help=(
            "Fidelity |⟨ψ_i|ψ_j⟩|² threshold for clustering.  Complete-linkage: "
            "every pair of circuits within a cluster must individually meet this "
            "threshold.  Default: 0.999."
        ),
    )
    parser.add_argument(
        "--rng-seed",
        type=int,
        default=42,
        help="Global RNG seed.  Default: 42.",
    )
    return parser.parse_args()


# ── Gate string parsing ────────────────────────────────────────────────────────

_AXIS_MAP = {"RX": 1, "RY": 2, "RZ": 3}
_GATE_ROT = re.compile(r"(RX|RY|RZ)\(q=(\d+)\)")
_GATE_CNOT = re.compile(r"CNOT\((\d+)->(\d+)\)")


def parse_gate_string(s: str) -> list[GateSpec]:
    """Parse 'RY(q=2) | CNOT(2->5) | RX(q=3) | ...' into GateSpec objects.

    Warns for any non-empty token that matches neither a rotation nor a CNOT,
    so truncated or extended formats are visible rather than silently dropped.
    """
    specs: list[GateSpec] = []
    for i, token in enumerate(s.split("|")):
        token = token.strip()
        if not token:
            continue
        m = _GATE_ROT.fullmatch(token)
        if m:
            specs.append(GateSpec(
                gate_type="rot", action_id=i, step_index=i,
                q=int(m.group(2)), axis=_AXIS_MAP[m.group(1)], angle=0.0,
            ))
            continue
        m = _GATE_CNOT.fullmatch(token)
        if m:
            specs.append(GateSpec(
                gate_type="cnot", action_id=i, step_index=i,
                ctrl=int(m.group(1)), targ=int(m.group(2)),
            ))
            continue
        print(f"      WARNING: unrecognised gate token '{token}' at position {i} — skipped.")
    return specs


# ── Circuit optimization and state-vector extraction ──────────────────────────

def optimize_and_get_vec(
    gates: list[GateSpec],
    n_qubits: int,
    H: np.ndarray,
    energy_shift: float,
    exact_energy: float,
    optimizer: str,
    n_restarts: int,
    cobyla_maxiter: int,
    rotosolve_sweeps: int,
    rng: np.random.Generator,
) -> tuple[float, np.ndarray]:
    """Optimize circuit angles and return (error_mha, state_vector).

    For COBYLA: delegates n_restarts directly to circuit_utils.optimize_circuit.
    For Rotosolve: runs n_restarts independent random initializations, each
    followed by rotosolve_sweeps sweeps, and picks the best result.
    Multi-restart is essential for rotosolve because it is coordinate descent
    and can get trapped in local minima depending on starting angles.
    """
    n_rot = sum(1 for g in gates if g.gate_type == "rot")

    if optimizer != "rotosolve" or n_rot == 0:
        # COBYLA path: n_restarts handled inside circuit_utils.optimize_circuit.
        energy, opt_gates = _optimize_circuit(
            gates, n_qubits, H, energy_shift,
            optimizer=optimizer,
            maxiter=cobyla_maxiter,
            n_restarts=n_restarts,
            rotosolve_sweeps=rotosolve_sweeps,
            rng=rng,
        )
    else:
        # Rotosolve path: run n_restarts random starts, take the best.
        best_energy = float("inf")
        best_opt_gates: list[GateSpec] = gates
        for _ in range(max(1, n_restarts)):
            init = rng.uniform(-np.pi, np.pi, n_rot)
            energy, opt_gates = _optimize_circuit(
                gates, n_qubits, H, energy_shift,
                optimizer="rotosolve",
                init_angles=init,
                rotosolve_sweeps=rotosolve_sweeps,
                rng=rng,
            )
            if energy < best_energy:
                best_energy = energy
                best_opt_gates = opt_gates
        energy, opt_gates = best_energy, best_opt_gates

    qc = build_qulacs_circuit(opt_gates, n_qubits)
    st = QuantumState(n_qubits)
    st.set_zero_state()
    qc.update_quantum_state(st)
    vec = st.get_vector()
    error_mha = abs(energy - exact_energy) * 1000.0
    return error_mha, vec


# ── Clustering (deterministic complete-linkage agglomerative) ──────────────────

def assign_clusters(
    fmat: np.ndarray,
    threshold: float,
    item_keys: list[str] | None = None,
) -> list[int]:
    """Deterministic complete-linkage clustering on a precomputed fidelity matrix.

    Start from singleton clusters and repeatedly merge the pair whose minimum
    cross-cluster fidelity is largest, provided that minimum is still above the
    threshold.  Ties are broken lexicographically by stable item keys so the
    result is deterministic even if summary.tsv row order changes.

    This guarantees that every pair inside a reported cluster individually
    satisfies the fidelity threshold.
    """
    n = fmat.shape[0]
    clusters: list[list[int]] = [[i] for i in range(n)]
    eps = 1e-12
    keys = item_keys or [str(i) for i in range(n)]

    def cluster_key(cluster: list[int]) -> tuple[str, ...]:
        return tuple(sorted(keys[idx] for idx in cluster))

    def cluster_similarity(left: list[int], right: list[int]) -> float:
        return min(fmat[i, j] for i in left for j in right)

    while True:
        best_pair: tuple[int, int] | None = None
        best_score = float("-inf")
        best_key: tuple[tuple[int, ...], tuple[int, ...]] | None = None

        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                score = cluster_similarity(clusters[i], clusters[j])
                if score + eps < threshold:
                    continue
                pair_key = (cluster_key(clusters[i]), cluster_key(clusters[j]))
                if (
                    best_pair is None
                    or score > best_score + eps
                    or (abs(score - best_score) <= eps and pair_key < best_key)
                ):
                    best_pair = (i, j)
                    best_score = score
                    best_key = pair_key

        if best_pair is None:
            break

        i, j = best_pair
        merged = sorted(clusters[i] + clusters[j])
        new_clusters: list[list[int]] = []
        for idx, cluster in enumerate(clusters):
            if idx in {i, j}:
                continue
            new_clusters.append(cluster)
        new_clusters.append(merged)
        new_clusters.sort(key=cluster_key)
        clusters = new_clusters

    labels = [0] * n
    for cluster_id, members in enumerate(clusters):
        for idx in members:
            labels[idx] = cluster_id
    return labels


# ── TSV I/O ────────────────────────────────────────────────────────────────────

def read_summary_tsv(tsv_path: Path) -> list[dict]:
    with tsv_path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        return [dict(row) for row in reader]


def write_tsv(path: Path, header: list[str], rows: list[list]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(header)
        for row in rows:
            writer.writerow(row)


# ── Main analysis per directory ────────────────────────────────────────────────

def _short_label(episode_key: str, fallback: str) -> str:
    """Shorten an episode key to 'ep<N>_s<M>'."""
    m = re.search(r"ep(\d+)__snap(\d+)$", episode_key)
    if m:
        return f"ep{m.group(1)}_s{m.group(2)}"
    return fallback


def analyze_dir(
    analysis_dir: Path,
    args: argparse.Namespace,
    rng: np.random.Generator,
) -> None:
    tsv_path = analysis_dir / "summary.tsv"
    if not tsv_path.exists():
        print(f"  [skip] no summary.tsv in {analysis_dir}")
        return

    rows = read_summary_tsv(tsv_path)
    if not rows:
        print(f"  [skip] empty summary.tsv in {analysis_dir}")
        return

    # ── Cache RunContext and mol data, keyed by run_dir string ─────────────────
    ctx_cache: dict[str, RunContext] = {}
    mol_cache: dict[Path, tuple] = {}  # keyed by resolved mol_path

    def get_ctx(run_dir_str: str) -> RunContext:
        if run_dir_str not in ctx_cache:
            ctx_cache[run_dir_str] = build_run_context(Path(run_dir_str))
        return ctx_cache[run_dir_str]

    def get_mol(mol_path: Path) -> tuple[np.ndarray, float, float, int]:
        if mol_path not in mol_cache:
            data = np.load(str(mol_path), allow_pickle=True)
            H = data["hamiltonian"].astype(np.complex128)
            energy_shift = float(data.get("energy_shift", 0.0))
            exact_energy = float(np.min(data["eigvals"])) + energy_shift
            n_qubits = int(round(np.log2(H.shape[0])))
            mol_cache[mol_path] = (H, energy_shift, exact_energy, n_qubits)
        return mol_cache[mol_path]

    # ── Pre-flight: verify all rows target the same molecule ───────────────────
    mol_paths: set[Path] = set()
    for row in rows:
        run_dir_str = row.get("run_dir", "").strip()
        if not run_dir_str:
            continue
        try:
            ctx = get_ctx(run_dir_str)
            mol_paths.add(ctx.mol_path.resolve())
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load RunContext for run_dir '{run_dir_str}': {exc}"
            ) from exc

    if len(mol_paths) > 1:
        raise RuntimeError(
            f"summary.tsv in '{analysis_dir.name}' references {len(mol_paths)} "
            f"different molecules — fidelity comparison across different Hilbert "
            f"spaces is undefined.  Paths found:\n"
            + "\n".join(f"  {p}" for p in sorted(mol_paths))
        )

    print(f"  Processing {len(rows)} circuit(s) in {analysis_dir.name} …")

    labels: list[str] = []
    vecs: list[np.ndarray] = []
    errors_mha: list[float] = []
    gate_counts: list[int] = []
    n_rot_counts: list[int] = []
    optimizers_used: list[str] = []
    opt_params_used: list[str] = []   # human-readable hyperparams for the report

    for i, row in enumerate(rows):
        gate_str = row.get("retained_gates", "").strip()
        if not gate_str:
            continue

        run_dir_str = row.get("run_dir", "").strip()
        ctx = get_ctx(run_dir_str)
        H, energy_shift, exact_energy, n_qubits = get_mol(ctx.mol_path.resolve())

        gates = parse_gate_string(gate_str)
        if not gates:
            print(f"    [{i+1}/{len(rows)}] WARNING: no gates parsed from '{gate_str}', skipping.")
            continue

        n_rot = sum(1 for g in gates if g.gate_type == "rot")
        label = _short_label(row.get("episode_key", ""), fallback=f"circuit_{i}")

        # ── Resolve optimizer and hyperparams ──────────────────────────────────
        if args.optimizer == "inherit":
            resolved_opt = row.get("resolved_optimizer", "cobyla").strip().lower()
            # Inherit training hyperparams from config_used.cfg where available.
            if resolved_opt == "cobyla":
                eff_maxiter = ctx.train_cobyla_maxiter or args.cobyla_maxiter
                eff_sweeps = args.rotosolve_sweeps  # unused for cobyla
            else:
                eff_sweeps = ctx.train_rotosolve_sweeps or args.rotosolve_sweeps
                eff_maxiter = args.cobyla_maxiter    # unused for rotosolve
        else:
            resolved_opt = args.optimizer
            eff_maxiter = args.cobyla_maxiter
            eff_sweeps = args.rotosolve_sweeps

        param_desc = (
            f"maxiter={eff_maxiter}" if resolved_opt == "cobyla"
            else f"sweeps={eff_sweeps}"
        )

        labels.append(label)
        gate_counts.append(len(gates))
        n_rot_counts.append(n_rot)
        optimizers_used.append(resolved_opt)
        opt_params_used.append(param_desc)

        print(
            f"    [{i+1}/{len(rows)}] {label}: {len(gates)} gates ({n_rot} rot) "
            f"[{resolved_opt},{param_desc},restarts={args.n_restarts}] … ",
            end="",
            flush=True,
        )
        error_mha, vec = optimize_and_get_vec(
            gates, n_qubits, H, energy_shift, exact_energy,
            optimizer=resolved_opt,
            n_restarts=args.n_restarts,
            cobyla_maxiter=eff_maxiter,
            rotosolve_sweeps=eff_sweeps,
            rng=rng,
        )
        errors_mha.append(error_mha)
        vecs.append(vec)
        print(f"err = {error_mha:.4f} mHa")

    if not vecs:
        print(f"  [skip] no valid circuits in {analysis_dir.name}")
        return

    n = len(vecs)

    # Pairwise fidelity matrix.
    fmat = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            fmat[i, j] = abs(vecs[i].conj() @ vecs[j]) ** 2

    # Complete-linkage clustering.
    clusters = assign_clusters(fmat, args.fidelity_threshold, item_keys=labels)
    n_clusters = max(clusters) + 1

    # ── Write fidelity_matrix.tsv ──────────────────────────────────────────────
    write_tsv(
        analysis_dir / "fidelity_matrix.tsv",
        [""] + labels,
        [[labels[i]] + [f"{fmat[i, j]:.6f}" for j in range(n)] for i in range(n)],
    )

    # ── Write fidelity_analysis.md ─────────────────────────────────────────────
    md: list[str] = [
        "# Fidelity Analysis",
        "",
        f"- circuits analyzed: `{n}`",
        f"- fidelity threshold: `{args.fidelity_threshold}` (deterministic complete-linkage)",
        f"- optimizer: `{args.optimizer}`  |  n_restarts: `{args.n_restarts}`",
        f"- cobyla_maxiter fallback: `{args.cobyla_maxiter}`  |  rotosolve_sweeps fallback: `{args.rotosolve_sweeps}`",
        f"- distinct re-optimized state clusters: **{n_clusters}**",
        "",
        "> **What this measures**: states produced by re-optimizing each retained",
        "> circuit structure independently from scratch.  n_clusters == 1 means",
        "> all CNOT skeletons fall into one high-fidelity state family at the chosen",
        "> threshold after re-optimization.  It does NOT prove identical variational",
        "> basins in the original training, nor exact physical identity beyond that",
        "> threshold.",
        "",
        "## Per-Circuit Results",
        "",
        "| # | Label | Gates | Rot | Optimizer | Params | Error (mHa) | Cluster |",
        "|--:|-------|------:|----:|:----------|:-------|------------:|:-------:|",
    ]
    for i, (lab, err, ng, nr, opt, par, cl) in enumerate(
        zip(labels, errors_mha, gate_counts, n_rot_counts,
            optimizers_used, opt_params_used, clusters)
    ):
        md.append(f"| {i} | `{lab}` | {ng} | {nr} | {opt} | {par} | {err:.4f} | {cl} |")

    # Fidelity matrix (Markdown table).
    md += [
        "",
        "## Pairwise Fidelity Matrix",
        "",
        "Values: `|⟨ψᵢ|ψⱼ⟩|²`.  Off-diagonal values ≥ threshold are **bold**.",
        "",
        "| |" + "".join(f" `{lab}` |" for lab in labels),
        "|:---|" + ":---:|" * n,
    ]
    for i, lab in enumerate(labels):
        cells = []
        for j in range(n):
            v = fmat[i, j]
            if i != j and v >= args.fidelity_threshold:
                cells.append(f"**{v:.4f}**")
            else:
                cells.append(f"{v:.4f}")
        md.append(f"| `{lab}` | " + " | ".join(cells) + " |")

    # Cluster summary.
    md += [
        "",
        "## Cluster Summary",
        "",
        f"Deterministic complete-linkage clusters at fidelity ≥ {args.fidelity_threshold}: **{n_clusters}**",
        "",
    ]
    for cl_id in range(n_clusters):
        members = [labels[i] for i, c in enumerate(clusters) if c == cl_id]
        mean_err = np.mean([errors_mha[i] for i, c in enumerate(clusters) if c == cl_id])
        md.append(
            f"**Cluster {cl_id}** ({len(members)} circuit(s), mean re-opt err {mean_err:.4f} mHa): "
            + ", ".join(f"`{m}`" for m in members)
        )

    md.append("")
    if n_clusters == 1:
        min_offdiag = min(
            fmat[i, j]
            for i in range(n) for j in range(n)
            if i != j
        ) if n > 1 else 1.0
        md += [
            f"> **Finding**: When re-optimized from scratch, all retained structures",
            f"> fall into one fidelity cluster at threshold {args.fidelity_threshold}",
            f"> (minimum pairwise fidelity inside the cluster = {min_offdiag:.4f}).",
            "> Structural diversity in summary.md is therefore consistent with",
            "> multiple retained circuits realizing the same re-optimized state",
            "> family, but this alone does not prove identical original basins.",
        ]
    else:
        min_cross = min(
            fmat[i, j]
            for i in range(n) for j in range(n)
            if clusters[i] != clusters[j]
        )
        md += [
            f"> **Finding**: {n_clusters} distinct re-optimized state cluster(s)",
            f"> detected (minimum cross-cluster fidelity = {min_cross:.4f}).",
            "> Possible explanations:",
            "> (a) Genuine near-degeneracy — multiple variational solutions at the",
            ">     same energy level (check cluster energy errors for equality).",
            "> (b) Expressibility gap — some CNOT skeletons cannot reach the ground",
            ">     state at all (check whether high-error circuits form their own cluster).",
            "> (c) Insufficient restarts — increase --n-restarts to rule out (c) first.",
        ]

    (analysis_dir / "fidelity_analysis.md").write_text(
        "\n".join(md) + "\n", encoding="utf-8"
    )
    print(
        f"    → {n_clusters} cluster(s) | "
        f"wrote fidelity_analysis.md + fidelity_matrix.tsv"
    )


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> int:
    args = parse_args()
    rng = np.random.default_rng(args.rng_seed)

    targets: list[Path] = []
    for raw in args.inputs:
        path = Path(raw).expanduser().resolve()
        if not path.exists():
            print(f"[skip] path does not exist: {path}")
            continue
        if path.is_file() and path.name == "summary.tsv":
            targets.append(path.parent)
        elif path.is_dir() and (path / "summary.tsv").exists():
            targets.append(path)
        elif path.is_dir():
            found = sorted(p.parent for p in path.rglob("summary.tsv"))
            if not found:
                print(f"[skip] no summary.tsv found under {path}")
                continue
            targets.extend(found)
        else:
            print(f"[skip] {path}: not a directory with summary.tsv")

    if not targets:
        print("No analysis directories found.")
        return 1

    for target in targets:
        print(f"\n[{target.name}]")
        analyze_dir(target, args, rng)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
