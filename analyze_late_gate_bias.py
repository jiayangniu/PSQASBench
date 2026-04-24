#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from critical_structure_tool.io_utils import (
    action_id_from_item,
    discover_run_dirs,
    build_run_context,
    parse_episode_traces,
)


REPO_ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze gate-placement bias in the last fraction of RLQAS training episodes. "
            "The script reports average per-episode usage of rotations, CNOT edges, and qubit roles."
        )
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        help=(
            "One or more run directories (seed11111 style) or higher-level result roots. "
            "The script recursively discovers episode_traces.txt files."
        ),
    )
    parser.add_argument(
        "--late-fraction",
        type=float,
        default=1.0 / 3.0,
        help="Use the last fraction of training episodes. Default: 1/3.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=15,
        help="How many top entries to show per category in summary.md. Default: 15.",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help=(
            "Optional output directory. If omitted, the script auto-generates one under "
            "gate_bias_analysis/."
        ),
    )
    return parser.parse_args()


def _slug(text: str) -> str:
    keep = []
    for ch in str(text):
        if ch.isalnum():
            keep.append(ch.lower())
        elif ch in {"-", "_", "."}:
            keep.append(ch)
        else:
            keep.append("_")
    slug = "".join(keep).strip("_")
    while "__" in slug:
        slug = slug.replace("__", "_")
    return slug or "unknown"


def _compact_molecule_token(name: str) -> str:
    parts = [part for part in str(name).split("_") if part]
    if not parts:
        return "unknown"
    keep = [parts[0], parts[1] if len(parts) > 1 else parts[0]]
    q_token = next((part for part in reversed(parts) if part.lower().endswith("q")), None)
    if q_token is not None and q_token not in keep:
        keep.append(q_token)
    return _slug("_".join(keep))


def _config_suffix_token(config_name: str, molecule_name: str) -> str:
    config = str(config_name)
    prefix = f"{molecule_name}_"
    if config.startswith(prefix):
        config = config[len(prefix):]
    return _slug(config)


def auto_out_dir(run_contexts: list) -> Path:
    methods = sorted({ctx.method for ctx in run_contexts})
    molecules = sorted({ctx.molecule for ctx in run_contexts})
    configs = sorted({ctx.config_name for ctx in run_contexts})

    method_token = _slug(methods[0]) if len(methods) == 1 else "multi_method"
    molecule_token = _compact_molecule_token(molecules[0]) if len(molecules) == 1 else "multi_molecule"
    if len(configs) == 1 and len(molecules) == 1:
        config_token = _config_suffix_token(configs[0], molecules[0])
    elif len(configs) == 1:
        config_token = _slug(configs[0])
    else:
        config_token = "multi_config"

    name = "__".join([method_token, molecule_token, config_token, "late_gate_bias"])
    return (REPO_ROOT / "gate_bias_analysis" / name).resolve()


def auto_figure_dir(out_dir: Path) -> Path:
    return (REPO_ROOT / "figure" / "analyze_late_gate_bias" / out_dir.name).resolve()


def write_tsv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def edge_token(action_id: int, action_dict: dict[int, list[int]], num_qubits: int) -> tuple[str, int, int] | None:
    ctrl, offset, rot_qubit, _rot_axis = action_dict[action_id]
    if rot_qubit < num_qubits:
        return None
    targ = (ctrl + offset) % num_qubits
    return f"CNOT({ctrl}->{targ})", ctrl, targ


def rot_token(action_id: int, action_dict: dict[int, list[int]], num_qubits: int) -> tuple[str, str, int] | None:
    _ctrl, _offset, rot_qubit, rot_axis = action_dict[action_id]
    if rot_qubit >= num_qubits:
        return None
    axis_name = {1: "RX", 2: "RY", 3: "RZ"}.get(rot_axis, f"R{rot_axis}")
    return f"{axis_name}(q={rot_qubit})", axis_name, rot_qubit


def mean_episode_index(selected_eps: list[dict]) -> float:
    if not selected_eps:
        return 0.0
    return sum(int(ep["episode"]) for ep in selected_eps) / len(selected_eps)


def plot_heatmap(
    matrix: np.ndarray,
    xlabels: list[str],
    ylabels: list[str],
    title: str,
    path: Path,
    cmap: str = "YlOrRd",
    fmt: str = ".2f",
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig_w = max(6.0, 0.9 * len(xlabels))
    fig_h = max(3.2, 0.9 * len(ylabels))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(matrix, cmap=cmap, aspect="auto")
    ax.set_xticks(range(len(xlabels)))
    ax.set_xticklabels(xlabels, rotation=0)
    ax.set_yticks(range(len(ylabels)))
    ax.set_yticklabels(ylabels)
    ax.set_title(title, pad=14, fontsize=14, fontweight="semibold")
    ax.set_xlabel("Target qubit", labelpad=10)
    ax.set_ylabel("Source qubit", labelpad=10)
    cbar = fig.colorbar(im, ax=ax)
    cbar.ax.set_ylabel("Avg count / episode", rotation=90)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            val = matrix[i, j]
            if abs(val) < 1e-12:
                continue
            ax.text(j, i, format(val, fmt), ha="center", va="center", fontsize=9, color="black")

    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_qubit_bias_bars(
    qubit_labels: list[str],
    avg_gate: list[float],
    avg_rot: list[float],
    avg_cnot: list[float],
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    x = np.arange(len(qubit_labels))
    width = 0.56

    fig, ax = plt.subplots(figsize=(8.6, 4.9))
    colors = {
        "rot": "#F58518",
        "cnot": "#54A24B",
        "total": "#1F4E79",
    }

    rot_bars = ax.bar(x, avg_rot, width=width, color=colors["rot"], label="Avg rot gate")
    cnot_bars = ax.bar(
        x,
        avg_cnot,
        width=width,
        bottom=avg_rot,
        color=colors["cnot"],
        label="Avg CNOT gate",
    )
    total_line = ax.plot(
        x,
        avg_gate,
        color=colors["total"],
        linewidth=2.2,
        marker="o",
        markersize=5.5,
        label="Avg gate",
        zorder=5,
    )[0]

    ax.set_xticks(x)
    ax.set_xticklabels(qubit_labels)
    ax.set_ylabel("Avg count / episode")
    ax.set_xlabel("Qubit")
    ax.set_title("Late-Training Gate Placement Bias by Qubit", pad=18, fontsize=14, fontweight="semibold")
    ax.grid(axis="y", linestyle="--", linewidth=0.8, alpha=0.35)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(
        frameon=False,
        handles=[total_line, rot_bars[0], cnot_bars[0]],
        labels=["Avg gate", "Avg rot gate", "Avg CNOT gate"],
        ncol=3,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        borderaxespad=0.0,
        columnspacing=1.4,
        handletextpad=0.5,
    )

    ymax = max(max(avg_gate), max(avg_rot), max(avg_cnot), 1e-6)
    ax.set_ylim(0.0, ymax * 1.18)

    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.96])
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    args = parse_args()
    run_dirs = discover_run_dirs(args.inputs)
    if not run_dirs:
        raise SystemExit("No valid run directories with episode_traces.txt were found.")

    run_contexts = [build_run_context(run_dir) for run_dir in run_dirs]
    out_dir = (
        Path(args.out_dir).expanduser().resolve()
        if args.out_dir is not None
        else auto_out_dir(run_contexts)
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    figure_dir = auto_figure_dir(out_dir)
    figure_dir.mkdir(parents=True, exist_ok=True)

    aggregate_counts = {
        "gate_token": Counter(),
        "rot_token": Counter(),
        "edge_token": Counter(),
        "rot_axis": Counter(),
        "rot_qubit": Counter(),
        "cnot_src": Counter(),
        "cnot_targ": Counter(),
        "qubit_touch": Counter(),
    }
    presence_counts = {name: Counter() for name in aggregate_counts}

    episode_rows = []
    selected_total = 0
    total_actions = 0
    total_rot = 0
    total_cnot = 0

    for ctx in run_contexts:
        episodes = parse_episode_traces(ctx.run_dir / "episode_traces.txt")
        if not episodes:
            continue
        min_episode = int(len(episodes) * (1.0 - args.late_fraction))
        selected_eps = [ep for ep in episodes if int(ep["episode"]) >= min_episode]
        if not selected_eps:
            selected_eps = episodes

        for ep in selected_eps:
            episode_index = int(ep["episode"])
            actions = ep.get("actions", [])
            action_ids = [action_id_from_item(item) for item in actions]

            per_episode = {name: Counter() for name in aggregate_counts}
            rot_count = 0
            cnot_count = 0
            qubit_touch_set = set()

            for action_id in action_ids:
                rot_info = rot_token(action_id, ctx.action_dict, ctx.num_qubits)
                if rot_info is not None:
                    token, axis_name, q = rot_info
                    per_episode["gate_token"][token] += 1
                    per_episode["rot_token"][token] += 1
                    per_episode["rot_axis"][axis_name] += 1
                    per_episode["rot_qubit"][f"q{q}"] += 1
                    per_episode["qubit_touch"][f"q{q}"] += 1
                    qubit_touch_set.add(f"q{q}")
                    rot_count += 1
                    continue

                edge_info = edge_token(action_id, ctx.action_dict, ctx.num_qubits)
                if edge_info is None:
                    continue
                edge, src, targ = edge_info
                per_episode["gate_token"][edge] += 1
                per_episode["edge_token"][edge] += 1
                per_episode["cnot_src"][f"q{src}"] += 1
                per_episode["cnot_targ"][f"q{targ}"] += 1
                per_episode["qubit_touch"][f"q{src}"] += 1
                per_episode["qubit_touch"][f"q{targ}"] += 1
                qubit_touch_set.add(f"q{src}")
                qubit_touch_set.add(f"q{targ}")
                cnot_count += 1

            for category, counter in per_episode.items():
                aggregate_counts[category].update(counter)
                for token in counter:
                    presence_counts[category][token] += 1

            selected_total += 1
            total_actions += len(action_ids)
            total_rot += rot_count
            total_cnot += cnot_count

            episode_rows.append(
                {
                    "run_dir": str(ctx.run_dir),
                    "method": ctx.method,
                    "molecule": ctx.molecule,
                    "config": ctx.config_name,
                    "seed": ctx.seed_name,
                    "episode": episode_index,
                    "n_actions": len(action_ids),
                    "n_rot": rot_count,
                    "n_cnot": cnot_count,
                    "gate_token_counts": json.dumps(dict(per_episode["gate_token"]), ensure_ascii=True, sort_keys=True),
                    "rot_qubit_counts": json.dumps(dict(per_episode["rot_qubit"]), ensure_ascii=True, sort_keys=True),
                    "cnot_edge_counts": json.dumps(dict(per_episode["edge_token"]), ensure_ascii=True, sort_keys=True),
                    "cnot_src_counts": json.dumps(dict(per_episode["cnot_src"]), ensure_ascii=True, sort_keys=True),
                    "cnot_targ_counts": json.dumps(dict(per_episode["cnot_targ"]), ensure_ascii=True, sort_keys=True),
                    "qubit_touch_counts": json.dumps(dict(per_episode["qubit_touch"]), ensure_ascii=True, sort_keys=True),
                    "qubits_touched": json.dumps(sorted(qubit_touch_set), ensure_ascii=True),
                }
            )

    if selected_total == 0:
        raise SystemExit("No episodes were selected from the requested late-training window.")

    write_tsv(
        out_dir / "episode_gate_usage.tsv",
        [
            "run_dir",
            "method",
            "molecule",
            "config",
            "seed",
            "episode",
            "n_actions",
            "n_rot",
            "n_cnot",
            "gate_token_counts",
            "rot_qubit_counts",
            "cnot_edge_counts",
            "cnot_src_counts",
            "cnot_targ_counts",
            "qubit_touch_counts",
            "qubits_touched",
        ],
        episode_rows,
    )

    average_rows = []
    category_total_counts = {name: sum(counter.values()) for name, counter in aggregate_counts.items()}
    for category, counter in aggregate_counts.items():
        total_in_category = category_total_counts[category]
        for rank, (token, count) in enumerate(counter.most_common(), start=1):
            average_rows.append(
                {
                    "category": category,
                    "rank": rank,
                    "token": token,
                    "total_count": count,
                    "avg_count_per_episode": f"{count / selected_total:.6f}",
                    "episode_presence_frac": f"{presence_counts[category][token] / selected_total:.6f}",
                    "share_within_category": (
                        f"{count / total_in_category:.6f}" if total_in_category > 0 else "0.000000"
                    ),
                }
            )

    write_tsv(
        out_dir / "average_gate_usage.tsv",
        [
            "category",
            "rank",
            "token",
            "total_count",
            "avg_count_per_episode",
            "episode_presence_frac",
            "share_within_category",
        ],
        average_rows,
    )

    num_qubits = run_contexts[0].num_qubits if run_contexts else 0
    qubit_labels = [f"q{i}" for i in range(num_qubits)]
    rot_qubit_values = [
        aggregate_counts["rot_qubit"].get(f"q{i}", 0) / selected_total for i in range(num_qubits)
    ]
    touched_qubit_values = [
        aggregate_counts["qubit_touch"].get(f"q{i}", 0) / selected_total for i in range(num_qubits)
    ]
    cnot_src_values = [
        aggregate_counts["cnot_src"].get(f"q{i}", 0) / selected_total for i in range(num_qubits)
    ]
    cnot_targ_values = [
        aggregate_counts["cnot_targ"].get(f"q{i}", 0) / selected_total for i in range(num_qubits)
    ]
    cnot_qubit_values = [src + targ for src, targ in zip(cnot_src_values, cnot_targ_values)]

    cnot_matrix = np.zeros((num_qubits, num_qubits), dtype=float)
    for src in range(num_qubits):
        for targ in range(num_qubits):
            token = f"CNOT({src}->{targ})"
            cnot_matrix[src, targ] = aggregate_counts["edge_token"].get(token, 0) / selected_total

    for legacy_name in [
        "rotation_qubit_heatmap.png",
        "qubit_touch_heatmap.png",
        "cnot_source_heatmap.png",
        "cnot_target_heatmap.png",
    ]:
        legacy_path = figure_dir / legacy_name
        if legacy_path.exists():
            legacy_path.unlink()

    plot_qubit_bias_bars(
        qubit_labels,
        touched_qubit_values,
        rot_qubit_values,
        cnot_qubit_values,
        figure_dir / "qubit_bias_bars.png",
    )
    plot_heatmap(
        cnot_matrix,
        qubit_labels,
        qubit_labels,
        "Late-Training CNOT Edge Usage",
        figure_dir / "cnot_edge_heatmap.png",
        cmap="YlOrRd",
        fmt=".3f",
    )

    summary_lines = [
        "# Late-Training Gate Bias Summary",
        "",
        f"- discovered runs: `{len(run_dirs)}`",
        f"- selected episodes: `{selected_total}`",
        f"- late fraction: `{args.late_fraction}`",
        f"- average actions per episode: `{total_actions / selected_total:.3f}`",
        f"- average rotation gates per episode: `{total_rot / selected_total:.3f}`",
        f"- average CNOT gates per episode: `{total_cnot / selected_total:.3f}`",
        f"- figure directory: `{figure_dir}`",
        "",
        "This analysis averages gate placement over the last training fraction and is intended to reveal structural bias rather than isolate a single critical circuit.",
        "",
    ]

    sections = [
        ("Top Gate Tokens", "gate_token"),
        ("Top Rotation-Qubit Tokens", "rot_qubit"),
        ("Top CNOT Edge Tokens", "edge_token"),
        ("Top CNOT Source Qubits", "cnot_src"),
        ("Top CNOT Target Qubits", "cnot_targ"),
        ("Top Touched Qubits", "qubit_touch"),
    ]

    for title, category in sections:
        summary_lines += [f"## {title}", ""]
        summary_lines += [
            "| Rank | Token | Avg / Episode | Presence | Share |",
            "|---|---|---:|---:|---:|",
        ]
        for row in [r for r in average_rows if r["category"] == category][: args.top_k]:
            summary_lines.append(
                f"| {row['rank']} | `{row['token']}` | {row['avg_count_per_episode']} | "
                f"{row['episode_presence_frac']} | {row['share_within_category']} |"
            )
        summary_lines.append("")

    (out_dir / "summary.md").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    meta = {
        "late_fraction": args.late_fraction,
        "selected_episodes": selected_total,
        "average_actions_per_episode": total_actions / selected_total,
        "average_rotation_gates_per_episode": total_rot / selected_total,
        "average_cnot_gates_per_episode": total_cnot / selected_total,
        "run_dirs": [str(path) for path in run_dirs],
        "figure_dir": str(figure_dir),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"Saved late-training gate-bias analysis to: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
