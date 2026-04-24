#!/usr/bin/env python3
"""
plot_fidelity_cluster_progress.py

Read fidelity_analysis.md from one or more critical_structure_analysis
directories and generate:

1. cluster_assignments.tsv
2. cluster_cumulative.tsv
3. cluster_bins_<bin_size>.tsv
4. PNG plots for cumulative counts / fractions and binned counts / fractions

The intended workflow is:

1. run analyze_critical_structure.py on a bucket directory
2. run analyze_fidelity.py on that same directory
3. run this script to summarize how fidelity clusters evolve with episode index

Example:

    python plot_fidelity_cluster_progress.py \
      critical_structure_analysis/l3_ch2_8q_bucket0p00_ep10000_20000
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent

FIDELITY_ROW_RE = re.compile(
    r"^\|\s*(\d+)\s*\|\s*`?([^`|]+)`?\s*\|\s*(\d+)\s*\|\s*(\d+)\s*\|"
    r"\s*([^|]+?)\s*\|\s*([^|]+?)\s*\|\s*([0-9.eE+-]+)\s*\|\s*(\d+)\s*\|$"
)
EPISODE_KEY_RE = re.compile(r"ep(\d+)__snap(\d+)$")
SHORT_LABEL_RE = re.compile(r"ep(\d+)_s(\d+)$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate per-cluster assignment tables and training-progress plots "
            "from fidelity_analysis.md."
        )
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        help=(
            "One or more critical_structure_analysis directories. Each should "
            "contain fidelity_analysis.md produced by analyze_fidelity.py. "
            "If a parent directory is given, subdirectories are searched recursively."
        ),
    )
    parser.add_argument(
        "--bin-size",
        type=int,
        default=1000,
        help="Episode bin size for aggregated cluster counts. Default: 1000.",
    )
    parser.add_argument(
        "--out-root",
        default="figure",
        help=(
            "Root directory for generated outputs. A per-analysis subdirectory "
            "named after the input directory will be created under this root. "
            "Relative paths are resolved from the repo root. Default: figure"
        ),
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Only write TSV files; skip PNG figure generation.",
    )
    return parser.parse_args()


def find_analysis_dirs(inputs: list[str]) -> list[Path]:
    found: list[Path] = []
    for raw in inputs:
        path = Path(raw).resolve()
        if path.is_file() and path.name == "fidelity_analysis.md":
            found.append(path.parent)
            continue
        if path.is_dir():
            if (path / "fidelity_analysis.md").exists():
                found.append(path)
                continue
            subdirs = sorted(
                p.parent for p in path.rglob("fidelity_analysis.md")
                if (p.parent / "fidelity_analysis.md").exists()
            )
            found.extend(subdirs)
            continue
        print(f"[skip] {path}: not found", file=sys.stderr)
    deduped: list[Path] = []
    seen: set[Path] = set()
    for path in found:
        if path not in seen:
            deduped.append(path)
            seen.add(path)
    return deduped


def read_summary_tsv(tsv_path: Path) -> list[dict[str, str]]:
    with tsv_path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        return [dict(row) for row in reader]


def infer_label_from_summary(row: dict[str, str], fallback_index: int) -> str:
    episode_key = row.get("episode_key", "").strip()
    m = EPISODE_KEY_RE.search(episode_key)
    if m:
        return f"ep{m.group(1)}_s{m.group(2)}"
    ep = row.get("episode", "").strip()
    snap = row.get("snapshot_index", "").strip()
    if ep and snap:
        return f"ep{ep}_s{snap}"
    return f"summary_row_{fallback_index}"


def parse_episode_snapshot(label: str, fallback_index: int) -> tuple[int, int]:
    m = SHORT_LABEL_RE.fullmatch(label)
    if m:
        return int(m.group(1)), int(m.group(2))
    return fallback_index, 0


def parse_fidelity_markdown(md_path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    in_table = False
    with md_path.open(encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.rstrip("\n")
            if line.startswith("| # | Label |"):
                in_table = True
                continue
            if not in_table:
                continue
            if not line.startswith("|"):
                if rows:
                    break
                continue
            if line.startswith("|--"):
                continue
            m = FIDELITY_ROW_RE.match(line)
            if not m:
                continue
            label = m.group(2).strip()
            episode, snapshot_index = parse_episode_snapshot(label, int(m.group(1)))
            rows.append(
                {
                    "fidelity_row_index": int(m.group(1)),
                    "label": label,
                    "episode": episode,
                    "snapshot_index": snapshot_index,
                    "fidelity_gates": int(m.group(3)),
                    "fidelity_rot": int(m.group(4)),
                    "fidelity_optimizer": m.group(5).strip(),
                    "fidelity_params": m.group(6).strip(),
                    "fidelity_error_mha": float(m.group(7)),
                    "cluster": int(m.group(8)),
                }
            )
    if not rows:
        raise RuntimeError(f"Failed to parse per-circuit table from {md_path}")
    return rows


def merge_with_summary(
    fidelity_rows: list[dict[str, object]],
    summary_rows: list[dict[str, str]],
) -> tuple[list[dict[str, object]], list[str]]:
    if not summary_rows:
        merged = [dict(row) for row in fidelity_rows]
        return merged, []

    active_rows: list[dict[str, object]] = []
    for idx, row in enumerate(summary_rows):
        if not row.get("retained_gates", "").strip():
            continue
        augmented: dict[str, object] = dict(row)
        augmented["_summary_row_index"] = idx
        augmented["label"] = infer_label_from_summary(row, idx)
        try:
            augmented["episode"] = int(str(row.get("episode", "")).strip())
        except ValueError:
            episode, _ = parse_episode_snapshot(str(augmented["label"]), idx)
            augmented["episode"] = episode
        try:
            augmented["snapshot_index"] = int(str(row.get("snapshot_index", "")).strip())
        except ValueError:
            _, snap = parse_episode_snapshot(str(augmented["label"]), idx)
            augmented["snapshot_index"] = snap
        active_rows.append(augmented)

    label_map: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in active_rows:
        label_map[str(row["label"])].append(row)

    merged: list[dict[str, object]] = []
    warnings: list[str] = []
    used_summary_indices: set[int] = set()

    for i, fidelity_row in enumerate(fidelity_rows):
        chosen: dict[str, object] | None = None

        if i < len(active_rows):
            candidate = active_rows[i]
            if (
                str(candidate["label"]) == str(fidelity_row["label"])
                and int(candidate["_summary_row_index"]) not in used_summary_indices
            ):
                chosen = candidate

        if chosen is None:
            candidates = [
                row for row in label_map.get(str(fidelity_row["label"]), [])
                if int(row["_summary_row_index"]) not in used_summary_indices
            ]
            if len(candidates) == 1:
                chosen = candidates[0]

        if chosen is None and i < len(active_rows):
            chosen = active_rows[i]
            warnings.append(
                f"row {i}: label mismatch between fidelity '{fidelity_row['label']}' "
                f"and summary '{chosen['label']}', fell back to row-order merge"
            )

        if chosen is None:
            warnings.append(
                f"row {i}: no matching summary row found for fidelity label "
                f"'{fidelity_row['label']}'"
            )
            merged.append(dict(fidelity_row))
            continue

        used_summary_indices.add(int(chosen["_summary_row_index"]))

        combined: dict[str, object] = {}
        for key, value in chosen.items():
            if key == "_summary_row_index":
                continue
            combined[key] = value
        for key, value in fidelity_row.items():
            combined[key] = value
        merged.append(combined)

    return merged, warnings


def cluster_ids_from_rows(rows: list[dict[str, object]]) -> list[int]:
    return sorted({int(row["cluster"]) for row in rows})


def build_assignment_rows(rows: list[dict[str, object]]) -> tuple[list[str], list[list[object]]]:
    preferred = [
        "fidelity_row_index",
        "label",
        "episode",
        "snapshot_index",
        "cluster",
        "fidelity_error_mha",
        "fidelity_gates",
        "fidelity_rot",
        "fidelity_optimizer",
        "fidelity_params",
    ]
    all_keys: list[str] = []
    seen: set[str] = set()
    for key in preferred:
        if any(key in row for row in rows):
            all_keys.append(key)
            seen.add(key)
    for row in rows:
        for key in row.keys():
            if key not in seen:
                all_keys.append(key)
                seen.add(key)

    data_rows: list[list[object]] = []
    for row in rows:
        data_rows.append([row.get(key, "") for key in all_keys])
    return all_keys, data_rows


def build_cumulative_rows(
    rows: list[dict[str, object]],
    cluster_ids: list[int],
) -> tuple[list[str], list[list[object]], list[dict[str, object]]]:
    ordered = sorted(
        rows,
        key=lambda row: (
            int(row.get("episode", 0)),
            int(row.get("snapshot_index", 0)),
            int(row.get("fidelity_row_index", 0)),
        ),
    )
    counts = {cluster_id: 0 for cluster_id in cluster_ids}
    out_rows: list[dict[str, object]] = []

    for total_seen, row in enumerate(ordered, start=1):
        cluster = int(row["cluster"])
        counts[cluster] += 1
        out: dict[str, object] = {
            "episode": int(row.get("episode", 0)),
            "snapshot_index": int(row.get("snapshot_index", 0)),
            "label": row.get("label", ""),
            "cluster": cluster,
            "total_seen": total_seen,
        }
        for cluster_id in cluster_ids:
            cum = counts[cluster_id]
            out[f"cum_cluster{cluster_id}"] = cum
            out[f"frac_cluster{cluster_id}"] = cum / total_seen
        out_rows.append(out)

    header = ["episode", "snapshot_index", "label", "cluster", "total_seen"]
    header += [f"cum_cluster{cluster_id}" for cluster_id in cluster_ids]
    header += [f"frac_cluster{cluster_id}" for cluster_id in cluster_ids]
    data_rows = [[row.get(key, "") for key in header] for row in out_rows]
    return header, data_rows, out_rows


def build_binned_rows(
    rows: list[dict[str, object]],
    cluster_ids: list[int],
    bin_size: int,
) -> tuple[list[str], list[list[object]], list[dict[str, object]]]:
    bins: dict[int, dict[str, object]] = {}
    for row in rows:
        episode = int(row.get("episode", 0))
        cluster = int(row["cluster"])
        bin_start = (episode // bin_size) * bin_size
        bucket = bins.setdefault(
            bin_start,
            {
                "bin_start": bin_start,
                "bin_end": bin_start + bin_size - 1,
                "total": 0,
                "episode_min": episode,
                "episode_max": episode,
                **{f"count_cluster{cluster_id}": 0 for cluster_id in cluster_ids},
            },
        )
        bucket["total"] = int(bucket["total"]) + 1
        bucket["episode_min"] = min(int(bucket["episode_min"]), episode)
        bucket["episode_max"] = max(int(bucket["episode_max"]), episode)
        key = f"count_cluster{cluster}"
        bucket[key] = int(bucket[key]) + 1

    out_rows: list[dict[str, object]] = []
    for bin_start in sorted(bins):
        bucket = bins[bin_start]
        total = int(bucket["total"])
        for cluster_id in cluster_ids:
            count_key = f"count_cluster{cluster_id}"
            frac_key = f"frac_cluster{cluster_id}"
            count = int(bucket[count_key])
            bucket[frac_key] = (count / total) if total else 0.0
        out_rows.append(bucket)

    header = ["bin_start", "bin_end", "episode_min", "episode_max", "total"]
    header += [f"count_cluster{cluster_id}" for cluster_id in cluster_ids]
    header += [f"frac_cluster{cluster_id}" for cluster_id in cluster_ids]
    data_rows = [[row.get(key, "") for key in header] for row in out_rows]
    return header, data_rows, out_rows


def write_tsv(path: Path, header: list[str], rows: list[list[object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(header)
        writer.writerows(rows)


def resolve_output_dir(analysis_dir: Path, out_root_arg: str) -> Path:
    out_root = Path(out_root_arg)
    if not out_root.is_absolute():
        out_root = REPO_ROOT / out_root
    return out_root / analysis_dir.name


def _maybe_import_matplotlib():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - optional dependency path
        print(f"[warn] matplotlib unavailable, skipped plots: {exc}", file=sys.stderr)
        return None
    return plt


def plot_outputs(
    out_dir: Path,
    cluster_ids: list[int],
    cumulative_rows: list[dict[str, object]],
    binned_rows: list[dict[str, object]],
    bin_size: int,
) -> None:
    plt = _maybe_import_matplotlib()
    if plt is None:
        return

    cmap = plt.get_cmap("tab10")
    colors = {cluster_id: cmap(i % 10) for i, cluster_id in enumerate(cluster_ids)}

    if cumulative_rows:
        x = [int(row["episode"]) for row in cumulative_rows]

        fig, ax = plt.subplots(figsize=(8.5, 4.8))
        for cluster_id in cluster_ids:
            y = [float(row[f"cum_cluster{cluster_id}"]) for row in cumulative_rows]
            ax.plot(x, y, linewidth=2, label=f"Cluster {cluster_id}", color=colors[cluster_id])
        ax.set_xlabel("Episode")
        ax.set_ylabel("Cumulative sampled circuits")
        ax.set_title("Fidelity Cluster Cumulative Counts")
        ax.grid(True, alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / "cluster_cumulative_counts.png", dpi=200)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(8.5, 4.8))
        for cluster_id in cluster_ids:
            y = [float(row[f"frac_cluster{cluster_id}"]) for row in cumulative_rows]
            ax.plot(x, y, linewidth=2, label=f"Cluster {cluster_id}", color=colors[cluster_id])
        ax.set_xlabel("Episode")
        ax.set_ylabel("Cumulative fraction")
        ax.set_ylim(-0.02, 1.02)
        ax.set_title("Fidelity Cluster Cumulative Fractions")
        ax.grid(True, alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / "cluster_cumulative_fractions.png", dpi=200)
        plt.close(fig)

    if binned_rows:
        x = list(range(len(binned_rows)))
        labels = [f"{int(row['bin_start'])}-{int(row['bin_end'])}" for row in binned_rows]

        fig, ax = plt.subplots(figsize=(max(8.5, 0.9 * len(binned_rows)), 4.8))
        bottoms = [0.0] * len(binned_rows)
        for cluster_id in cluster_ids:
            y = [float(row[f"count_cluster{cluster_id}"]) for row in binned_rows]
            ax.bar(
                x,
                y,
                bottom=bottoms,
                label=f"Cluster {cluster_id}",
                color=colors[cluster_id],
                width=0.82,
            )
            bottoms = [bottom + val for bottom, val in zip(bottoms, y)]
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_xlabel(f"Episode bins (size = {bin_size})")
        ax.set_ylabel("Count")
        ax.set_title("Fidelity Cluster Counts by Episode Bin")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / f"cluster_bins_{bin_size}_counts.png", dpi=200)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(max(8.5, 0.9 * len(binned_rows)), 4.8))
        bottoms = [0.0] * len(binned_rows)
        for cluster_id in cluster_ids:
            y = [float(row[f"frac_cluster{cluster_id}"]) for row in binned_rows]
            ax.bar(
                x,
                y,
                bottom=bottoms,
                label=f"Cluster {cluster_id}",
                color=colors[cluster_id],
                width=0.82,
            )
            bottoms = [bottom + val for bottom, val in zip(bottoms, y)]
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_xlabel(f"Episode bins (size = {bin_size})")
        ax.set_ylabel("Fraction")
        ax.set_ylim(0.0, 1.0)
        ax.set_title("Fidelity Cluster Fractions by Episode Bin")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / f"cluster_bins_{bin_size}_fractions.png", dpi=200)
        plt.close(fig)


def process_dir(analysis_dir: Path, args: argparse.Namespace) -> None:
    md_path = analysis_dir / "fidelity_analysis.md"
    if not md_path.exists():
        print(f"[skip] {analysis_dir}: no fidelity_analysis.md", file=sys.stderr)
        return

    output_dir = resolve_output_dir(analysis_dir, args.out_root)
    output_dir.mkdir(parents=True, exist_ok=True)

    fidelity_rows = parse_fidelity_markdown(md_path)
    summary_path = analysis_dir / "summary.tsv"
    summary_rows = read_summary_tsv(summary_path) if summary_path.exists() else []
    merged_rows, warnings = merge_with_summary(fidelity_rows, summary_rows)
    cluster_ids = cluster_ids_from_rows(merged_rows)

    assignment_header, assignment_rows = build_assignment_rows(merged_rows)
    cumulative_header, cumulative_rows, cumulative_plot_rows = build_cumulative_rows(
        merged_rows, cluster_ids
    )
    binned_header, binned_rows, binned_plot_rows = build_binned_rows(
        merged_rows, cluster_ids, args.bin_size
    )

    write_tsv(output_dir / "cluster_assignments.tsv", assignment_header, assignment_rows)
    write_tsv(output_dir / "cluster_cumulative.tsv", cumulative_header, cumulative_rows)
    write_tsv(
        output_dir / f"cluster_bins_{args.bin_size}.tsv",
        binned_header,
        binned_rows,
    )

    if not args.no_plots:
        plot_outputs(
            output_dir,
            cluster_ids,
            cumulative_plot_rows,
            binned_plot_rows,
            args.bin_size,
        )

    print(f"[ok] {analysis_dir}")
    print(f"     output_dir: {output_dir}")
    print("     wrote cluster_assignments.tsv")
    print("     wrote cluster_cumulative.tsv")
    print(f"     wrote cluster_bins_{args.bin_size}.tsv")
    if not args.no_plots:
        print("     wrote cluster_cumulative_counts.png")
        print("     wrote cluster_cumulative_fractions.png")
        print(f"     wrote cluster_bins_{args.bin_size}_counts.png")
        print(f"     wrote cluster_bins_{args.bin_size}_fractions.png")
    for warning in warnings:
        print(f"     [warn] {warning}")


def main() -> None:
    args = parse_args()
    if args.bin_size <= 0:
        raise SystemExit("--bin-size must be positive")

    analysis_dirs = find_analysis_dirs(args.inputs)
    if not analysis_dirs:
        raise SystemExit("No analysis directories with fidelity_analysis.md found.")

    for analysis_dir in analysis_dirs:
        process_dir(analysis_dir, args)


if __name__ == "__main__":
    main()
