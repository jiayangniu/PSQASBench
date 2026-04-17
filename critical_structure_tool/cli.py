from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np

from .circuit_utils import prefix_to_gates
from .io_utils import (
    REPO_ROOT,
    choose_bucket_interactive,
    collect_episode_records,
    compute_anchor_action_counts,
    discover_run_dirs,
    sample_representative_episodes,
    summarize_buckets,
    summarize_first_hit_distribution,
)
from .pruning import compute_gate_importance, probabilistic_beam_prune
from .types import EpisodeRecord


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Mine critical circuit structure from multiple PSQASBench result directories "
            "using threshold-hit bucketing and probabilistic counterfactual pruning."
        )
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        help=(
            "One or more run directories (seed11111 style) or higher-level result roots. "
            "The tool recursively discovers episode_traces.txt files."
        ),
    )
    parser.add_argument(
        "--target-error-mha",
        type=float,
        default=None,
        help=(
            "Threshold used to define the first-hit circuit. "
            "If omitted, fallback to accept_err from run_meta.txt when available."
        ),
    )
    parser.add_argument(
        "--bucket",
        type=str,
        default=None,
        help="Bucket to analyze (mHa, rounded to 2 decimals). If omitted, prompt interactively.",
    )
    parser.add_argument(
        "--select-n",
        type=int,
        default=6,
        help="How many representative episodes to analyze. Default: 6.",
    )
    parser.add_argument(
        "--late-fraction",
        type=float,
        default=1.0 / 3.0,
        help="Prefer episodes from the last fraction of training within each run. Default: 1/3.",
    )
    parser.add_argument(
        "--anchor-top-k",
        type=int,
        default=3,
        help="Number of most frequent hit-time actions to mark as anchors. Default: 3.",
    )
    parser.add_argument(
        "--bucket-slack-mha",
        type=float,
        default=0.05,
        help="Allowed slack above the chosen bucket center during pruning. Default: 0.05 mHa.",
    )
    parser.add_argument(
        "--reconstruction-slack-mha",
        type=float,
        default=0.3,
        help=(
            "If reconstructed baseline is worse than the bucket center, allow this extra slack. "
            "Default: 0.3 mHa."
        ),
    )
    parser.add_argument(
        "--delta-tolerance-mha",
        type=float,
        default=0.2,
        help="Maximum allowed single-step error increase for a removable gate. Default: 0.2 mHa.",
    )
    parser.add_argument(
        "--beam-width",
        type=int,
        default=4,
        help="Beam width for probabilistic pruning. Default: 4.",
    )
    parser.add_argument(
        "--branching-factor",
        type=int,
        default=3,
        help="Beam-search top-k: how many deletion candidates to expand per state. Default: 3.",
    )
    parser.add_argument(
        "--prune-budget",
        type=int,
        default=1000,
        help=(
            "Global total budget for counterfactual child evaluations during pruning. "
            "Default: 1000."
        ),
    )
    parser.add_argument(
        "--max-prune-steps",
        type=int,
        default=None,
        help=(
            "Optional override for maximum pruning depth per episode. "
            "If omitted, compute automatically as gate_count - fixed_gate_count."
        ),
    )
    parser.add_argument(
        "--sampling-temperature-mha",
        type=float,
        default=0.1,
        help="Sampling temperature for probabilistic deletion. Smaller means greedier. Default: 0.1 mHa.",
    )
    parser.add_argument(
        "--protected-penalty",
        type=float,
        default=0.15,
        help="Probability penalty multiplier for anchor gates. Smaller means stronger protection. Default: 0.15.",
    )
    parser.add_argument(
        "--cobyla-maxiter",
        type=int,
        default=300,
        help="COBYLA max iterations for reconstruction/re-optimization. Default: 300.",
    )
    parser.add_argument(
        "--n-restarts",
        type=int,
        default=2,
        help="Number of COBYLA restarts per optimization. Default: 2.",
    )
    parser.add_argument(
        "--analysis-optimizer",
        choices=["inherit", "cobyla", "rotosolve"],
        default="inherit",
        help=(
            "Optimizer used inside the analysis tool for reconstruction and pruning. "
            "inherit -> use the optimizer family from config_used.cfg. Default: inherit."
        ),
    )
    parser.add_argument(
        "--rotosolve-sweeps",
        type=int,
        default=2,
        help="Number of sweeps when analysis optimizer is Rotosolve. Default: 2.",
    )
    parser.add_argument(
        "--rng-seed",
        type=int,
        default=123,
        help="Random seed for probabilistic beam pruning. Default: 123.",
    )
    parser.add_argument(
        "--out-dir",
        default="critical_structure_analysis",
        help="Directory to write all analysis outputs. Default: critical_structure_analysis",
    )
    return parser.parse_args()


def write_tsv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def load_molecule(path: Path):
    data = np.load(path, allow_pickle=True)
    H = data["hamiltonian"].astype(np.complex128)
    energy_shift = float(data.get("energy_shift", 0.0))
    exact_energy = float(np.min(data["eigvals"])) + energy_shift
    n_qubits = int(round(np.log2(H.shape[0])))
    return H, energy_shift, exact_energy, n_qubits


def episode_key(rec: EpisodeRecord) -> str:
    return (
        f"{rec.run.method}__{rec.run.molecule}__{rec.run.config_name}"
        f"__{rec.run.seed_name}__ep{rec.episode_index}"
    )


def format_gate_sequence(gates) -> str:
    return " | ".join(g.detailed_signature for g in gates)


def format_gate_signature_sequence(gates) -> str:
    return " | ".join(g.signature for g in gates)


def resolve_analysis_optimizer(requested: str, rec: EpisodeRecord) -> str:
    requested_norm = str(requested).lower()
    if requested_norm == "inherit":
        return rec.run.train_optimizer or "cobyla"
    return requested_norm


def resolve_cobyla_maxiter(args: argparse.Namespace, rec: EpisodeRecord, optimizer: str) -> int:
    if optimizer == "cobyla" and args.analysis_optimizer == "inherit" and rec.run.train_cobyla_maxiter:
        return int(rec.run.train_cobyla_maxiter)
    return int(args.cobyla_maxiter)


def resolve_rotosolve_sweeps(args: argparse.Namespace, rec: EpisodeRecord, optimizer: str) -> int:
    if optimizer == "rotosolve" and args.analysis_optimizer == "inherit" and rec.run.train_rotosolve_sweeps:
        return int(rec.run.train_rotosolve_sweeps)
    return int(args.rotosolve_sweeps)


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    run_dirs = discover_run_dirs(args.inputs)
    if not run_dirs:
        raise SystemExit("No valid run directories with episode_traces.txt were found.")

    print(f"Discovered {len(run_dirs)} run directories.")
    records = collect_episode_records(run_dirs, target_error_mha=args.target_error_mha)
    if not records:
        raise SystemExit("No episodes reached the requested target error threshold.")

    distribution_rows = summarize_first_hit_distribution(records)
    write_tsv(
        out_dir / "first_hit_error_distribution.tsv",
        [
            "display_value",
            "count",
            "seed_count",
            "min_first_hit_error_mha",
            "max_first_hit_error_mha",
        ],
        distribution_rows,
    )

    bucket_rows = summarize_buckets(records)
    write_tsv(
        out_dir / "bucket_summary.tsv",
        ["bucket", "count", "seed_count", "mean_hit_step", "mean_episode"],
        bucket_rows,
    )

    chosen_bucket = choose_bucket_interactive(bucket_rows, args.bucket)
    bucket_center_mha = float(chosen_bucket)
    bucket_records = [rec for rec in records if rec.bucket == chosen_bucket]

    anchor_counts = compute_anchor_action_counts(bucket_records)
    anchor_rows = [
        {"rank": i + 1, "action": action, "count": count}
        for i, (action, count) in enumerate(anchor_counts.most_common())
    ]
    write_tsv(out_dir / "anchor_actions.tsv", ["rank", "action", "count"], anchor_rows)

    protected_gate_signatures = {action for action, _ in anchor_counts.most_common(args.anchor_top_k)}
    rng = np.random.default_rng(args.rng_seed)

    selected = sample_representative_episodes(
        bucket_records,
        n_select=args.select_n,
        late_fraction=args.late_fraction,
        rng=rng,
    )
    if not selected:
        raise SystemExit("No representative episodes were selected.")

    selected_rows = []
    for rec in selected:
        selected_rows.append(
            {
                "episode_key": episode_key(rec),
                "run_dir": str(rec.run.run_dir),
                "method": rec.run.method,
                "molecule": rec.run.molecule,
                "config": rec.run.config_name,
                "seed": rec.run.seed_name,
                "episode": rec.episode_index,
                "first_hit_step": rec.first_hit_step,
                "first_hit_error_mha": f"{rec.first_hit_error_mha:.4f}",
                "last_action": rec.last_action_token,
                "prefix_len": len(rec.action_ids_prefix),
            }
        )
    write_tsv(
        out_dir / "selected_episodes.tsv",
        [
            "episode_key",
            "run_dir",
            "method",
            "molecule",
            "config",
            "seed",
            "episode",
            "first_hit_step",
            "first_hit_error_mha",
            "last_action",
            "prefix_len",
        ],
        selected_rows,
    )

    mol_cache = {}
    summary_rows = []
    exact_signature_counter: Counter[tuple[str, ...]] = Counter()
    retained_action_sets: list[set[str]] = []
    resolved_optimizers: Counter[str] = Counter()

    for idx, rec in enumerate(selected, start=1):
        if rec.run.mol_path not in mol_cache:
            mol_cache[rec.run.mol_path] = load_molecule(rec.run.mol_path)
        H, energy_shift, exact_energy, n_qubits = mol_cache[rec.run.mol_path]
        analysis_optimizer = resolve_analysis_optimizer(args.analysis_optimizer, rec)
        analysis_cobyla_maxiter = resolve_cobyla_maxiter(args, rec, analysis_optimizer)
        analysis_rotosolve_sweeps = resolve_rotosolve_sweeps(args, rec, analysis_optimizer)
        resolved_optimizers[analysis_optimizer] += 1

        initial_gates = prefix_to_gates(
            rec.action_ids_prefix,
            rec.run.action_dict,
            rec.run.num_qubits,
            first_hit_snapshot=rec.first_hit_snapshot,
        )
        print(
            f"[{idx}/{len(selected)}] {episode_key(rec)} "
            f"prefix_len={len(initial_gates)} qubits={n_qubits} optimizer={analysis_optimizer} "
            f"baseline+importance"
        )
        _baseline_energy, baseline_error_mha, initial_importance, baseline_gates = compute_gate_importance(
            initial_gates,
            n_qubits,
            H,
            energy_shift,
            exact_energy,
            protected_gate_signatures=protected_gate_signatures,
            maxiter=analysis_cobyla_maxiter,
            n_restarts=args.n_restarts,
            optimizer=analysis_optimizer,
            rotosolve_sweeps=analysis_rotosolve_sweeps,
            rng=rng,
        )

        allowed_error_mha = max(
            bucket_center_mha + args.bucket_slack_mha,
            baseline_error_mha + args.reconstruction_slack_mha,
        )
        fixed_gate_count = sum(
            1 for gate in baseline_gates if gate.signature in protected_gate_signatures
        )
        auto_max_prune_steps = max(0, len(baseline_gates) - fixed_gate_count)
        effective_max_prune_steps = auto_max_prune_steps
        if args.max_prune_steps is not None:
            effective_max_prune_steps = min(effective_max_prune_steps, int(args.max_prune_steps))

        print(
            f"[{idx}/{len(selected)}] {episode_key(rec)} "
            f"allowed_error_mha={allowed_error_mha:.6f} pruning "
            f"fixed_gates={fixed_gate_count} max_prune_steps={effective_max_prune_steps}"
        )
        best_branch, _beam_pool = probabilistic_beam_prune(
            baseline_gates,
            baseline_error_mha,
            initial_importance,
            n_qubits,
            H,
            energy_shift,
            exact_energy,
            allowed_error_mha=allowed_error_mha,
            delta_tolerance_mha=args.delta_tolerance_mha,
            protected_gate_signatures=protected_gate_signatures,
            beam_width=args.beam_width,
            branching_factor=args.branching_factor,
            max_prune_steps=effective_max_prune_steps,
            temperature_mha=args.sampling_temperature_mha,
            protected_penalty=args.protected_penalty,
            maxiter=analysis_cobyla_maxiter,
            n_restarts=args.n_restarts,
            optimizer=analysis_optimizer,
            rotosolve_sweeps=analysis_rotosolve_sweeps,
            prune_budget=args.prune_budget,
            rng=rng,
        )
        print(
            f"[{idx}/{len(selected)}] {episode_key(rec)} "
            f"done retained={len(best_branch.gates)}/{len(baseline_gates)} "
            f"delta_error_mha={best_branch.error_mha - baseline_error_mha:.6f}"
        )

        retained_labels = [g.signature for g in best_branch.gates]
        retained_compact = format_gate_signature_sequence(best_branch.gates)
        retained_action_sets.append(set(retained_labels))
        exact_signature_counter[tuple(retained_labels)] += 1
        original_gate_count = len(baseline_gates)
        retained_gate_count = len(best_branch.gates)
        removed_gate_count = original_gate_count - retained_gate_count
        redundancy_ratio = (
            removed_gate_count / original_gate_count if original_gate_count else 0.0
        )
        retained_error_delta_mha = best_branch.error_mha - baseline_error_mha
        summary_rows.append(
            {
                "episode_key": episode_key(rec),
                "run_dir": str(rec.run.run_dir),
                "episode": rec.episode_index,
                "first_hit_step": rec.first_hit_step,
                "first_hit_error_mha": f"{rec.first_hit_error_mha:.6f}",
                "baseline_error_mha": f"{baseline_error_mha:.6f}",
                "retained_error_mha": f"{best_branch.error_mha:.6f}",
                "delta_error_mha": f"{retained_error_delta_mha:.6f}",
                "allowed_error_mha": f"{allowed_error_mha:.6f}",
                "original_gate_count": original_gate_count,
                "fixed_gate_count": fixed_gate_count,
                "effective_max_prune_steps": effective_max_prune_steps,
                "retained_gate_count": retained_gate_count,
                "removed_gate_count": removed_gate_count,
                "redundancy_ratio_pct": f"{100.0 * redundancy_ratio:.2f}",
                "resolved_optimizer": analysis_optimizer,
                "last_action": rec.last_action_token,
                "retained_gates": retained_compact,
                "retained_signature": json.dumps(retained_labels, ensure_ascii=True),
            }
        )

    write_tsv(
        out_dir / "summary.tsv",
        [
            "episode_key",
            "run_dir",
            "episode",
            "first_hit_step",
            "first_hit_error_mha",
            "baseline_error_mha",
            "retained_error_mha",
            "delta_error_mha",
            "allowed_error_mha",
            "original_gate_count",
            "fixed_gate_count",
            "effective_max_prune_steps",
            "retained_gate_count",
            "removed_gate_count",
            "redundancy_ratio_pct",
            "resolved_optimizer",
            "last_action",
            "retained_gates",
            "retained_signature",
        ],
        summary_rows,
    )

    exact_rows = []
    for rank, (sig, count) in enumerate(exact_signature_counter.most_common(), 1):
        exact_rows.append(
            {
                "rank": rank,
                "count": count,
                "retained_gates": " | ".join(sig),
            }
        )
    write_tsv(out_dir / "exact_signature_counts.tsv", ["rank", "count", "retained_gates"], exact_rows)

    common_gate_signatures = []
    if retained_action_sets:
        common_gate_signatures = sorted(set.intersection(*retained_action_sets))

    meta_lines = [
        "analysis_type = critical_structure",
        f"target_error_mha = {'inherit_from_run_meta' if args.target_error_mha is None else f'{args.target_error_mha:.4f}'}",
        f"selected_bucket_mha = {chosen_bucket}",
        f"discovered_runs = {len(run_dirs)}",
        f"hit_episodes_in_bucket = {len(bucket_records)}",
        f"selected_episodes = {len(selected)}",
        f"late_fraction = {args.late_fraction}",
        f"analysis_optimizer_request = {args.analysis_optimizer}",
        f"rotosolve_sweeps = {args.rotosolve_sweeps}",
        f"anchor_top_k = {args.anchor_top_k}",
        f"beam_width = {args.beam_width}",
        f"branching_factor = {args.branching_factor}",
        f"max_prune_steps_override = {args.max_prune_steps if args.max_prune_steps is not None else 'auto'}",
        f"prune_budget = {args.prune_budget}",
        f"reconstruction_slack_mha = {args.reconstruction_slack_mha}",
        f"delta_tolerance_mha = {args.delta_tolerance_mha}",
        "",
        "[resolved_optimizers]",
    ]
    for optimizer_name, count in resolved_optimizers.items():
        meta_lines.append(f"{optimizer_name} = {count}")
    meta_lines += [
        "",
        "[anchor_actions]",
    ]
    for action, count in anchor_counts.most_common(args.anchor_top_k):
        meta_lines.append(f"{action} = {count}")
    meta_lines += ["", "[common_retained_gate_signatures]"]
    if common_gate_signatures:
        for sig in common_gate_signatures:
            meta_lines.append(sig)
    else:
        meta_lines.append("none")
    meta_lines += ["", "[exact_retained_structure_matches]"]
    if exact_rows:
        for row in exact_rows[:10]:
            meta_lines.append(f"count={row['count']} retained_gates={row['retained_gates']}")
    else:
        meta_lines.append("none")
    (out_dir / "meta.txt").write_text("\n".join(meta_lines) + "\n", encoding="utf-8")

    report_lines = [
        "# Summary",
        "",
        f"- target error threshold: `{'inherit_from_run_meta' if args.target_error_mha is None else f'{args.target_error_mha:.4f} mHa'}`",
        f"- selected bucket: `{chosen_bucket} mHa`",
        f"- discovered runs: `{len(run_dirs)}`",
        f"- hit episodes in bucket: `{len(bucket_records)}`",
        f"- selected episodes for pruning: `{len(selected)}`",
        "",
        "**Anchor Actions**",
        "",
    ]
    for action, count in anchor_counts.most_common(args.anchor_top_k):
        report_lines.append(f"- `{action}`: {count}")

    report_lines += [
        "",
        "**Per-Episode Pruning Summary**",
        "",
        "| Episode | Baseline (mHa) | Retained (mHa) | Δerror (mHa) | Original | Fixed | Max Steps | Retained | Removed | Redundancy | Optimizer | Retained Gates |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for row in summary_rows:
        report_lines.append(
            f"| `{row['episode_key']}` | {row['baseline_error_mha']} | "
            f"{row['retained_error_mha']} | {row['delta_error_mha']} | "
            f"{row['original_gate_count']} | {row['fixed_gate_count']} | {row['effective_max_prune_steps']} | "
            f"{row['retained_gate_count']} | {row['removed_gate_count']} | {row['redundancy_ratio_pct']}% | "
            f"{row['resolved_optimizer']} | `{row['retained_gates']}` |"
        )

    report_lines += [
        "",
        "**Exact Retained-Structure Matches**",
        "",
    ]
    if exact_rows:
        for row in exact_rows[:10]:
            report_lines.append(
                f"- count={row['count']}: `{row['retained_gates']}`"
            )
    else:
        report_lines.append("- none")

    report_lines += [
        "",
        "**Common Retained Gate Signatures**",
        "",
    ]
    if common_gate_signatures:
        for sig in common_gate_signatures:
            report_lines.append(f"- `{sig}`")
    else:
        report_lines.append("- none")

    (out_dir / "summary.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    print(f"\nSaved analysis outputs to: {out_dir}")
    print(f"Chosen bucket: {chosen_bucket} mHa")
    print(f"Selected episodes: {len(selected)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
