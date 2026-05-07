from __future__ import annotations

import ast
import csv
import html
import itertools
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parent
ANALYSIS_DIR = ROOT / "critical_structure_analysis"


@dataclass(frozen=True)
class FigureSpec:
    name: str
    title: str
    summary_tsv: Path
    representative_key: str
    cluster_keys: tuple[str, ...]
    cluster_label: str
    notes: str


SPECS = (
    FigureSpec(
        name="tier3_h2o",
        title="Tier3 H2O StrongCorr",
        summary_tsv=ANALYSIS_DIR / "tier3_h2o" / "summary.tsv",
        representative_key=(
            "crlqas__L4_H2O_StrongCorr_8q__"
            "L4_H2O_StrongCorr_8q_rotosolve_s2_20k__seed22222__ep4746__snap4"
        ),
        cluster_keys=(
            "crlqas__L4_H2O_StrongCorr_8q__"
            "L4_H2O_StrongCorr_8q_rotosolve_s2_20k__seed22222__ep4746__snap4",
            "crlqas__L4_H2O_StrongCorr_8q__"
            "L4_H2O_StrongCorr_8q_rotosolve_s2_20k__seed22222__ep4746__snap0",
            "crlqas__L4_H2O_StrongCorr_8q__"
            "L4_H2O_StrongCorr_8q_rotosolve_s2_20k__seed22222__ep4746__snap2",
            "crlqas__L4_H2O_StrongCorr_8q__"
            "L4_H2O_StrongCorr_8q_rotosolve_s2_20k__seed22222__ep4746__snap3",
            "crlqas__L4_H2O_StrongCorr_8q__"
            "L4_H2O_StrongCorr_8q_rotosolve_s2_20k__seed22222__ep4746__snap1",
        ),
        cluster_label="5-snapshot stable family from seed22222 / ep4746",
        notes=(
            "Representative circuit is the largest retained member of the stable "
            "family so the shared core and variable periphery are both visible."
        ),
    ),
    FigureSpec(
        name="tier3_h2_stretch",
        title="Tier3 H2 Stretch",
        summary_tsv=ANALYSIS_DIR / "tier3_h2_stretch" / "summary.tsv",
        representative_key=(
            "crlqas__L4_H2_Stretch_4q__"
            "L4_H2_Stretch_4q_cobyla_20k__seed33333__ep6934__snap0"
        ),
        cluster_keys=(
            "crlqas__L4_H2_Stretch_4q__"
            "L4_H2_Stretch_4q_cobyla_20k__seed33333__ep6934__snap0",
            "crlqas__L4_H2_Stretch_4q__"
            "L4_H2_Stretch_4q_cobyla_20k__seed33333__ep7772__snap0",
            "crlqas__L4_H2_Stretch_4q__"
            "L4_H2_Stretch_4q_cobyla_20k__seed33333__ep7974__snap0",
        ),
        cluster_label="3-snapshot stable family from seed33333",
        notes=(
            "No global core exists across all 10 retained circuits, so the figure "
            "focuses on the most internally consistent local family."
        ),
    ),
)


SINGLE_RE = re.compile(r"^(RX|RY|RZ)\(q=(\d+)\)$")
CNOT_RE = re.compile(r"^CNOT\((\d+)->(\d+)\)$")


def load_rows(path: Path) -> dict[str, dict[str, str]]:
    rows: dict[str, dict[str, str]] = {}
    with path.open() as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            row["sig_list"] = ast.literal_eval(row["retained_signature"])
            row["retained_gate_count_int"] = int(row["retained_gate_count"])
            rows[row["episode_key"]] = row
    return rows


def core_counter(cluster_rows: list[dict[str, str]]) -> Counter[str]:
    common = Counter(cluster_rows[0]["sig_list"])
    for row in cluster_rows[1:]:
        common &= Counter(row["sig_list"])
    return common


def highlighted_positions(sig_list: list[str], common: Counter[str]) -> set[int]:
    used = Counter()
    positions: set[int] = set()
    for idx, sig in enumerate(sig_list):
        if used[sig] < common[sig]:
            positions.add(idx)
            used[sig] += 1
    return positions


def infer_num_qubits(sig_list: list[str]) -> int:
    max_q = -1
    for sig in sig_list:
        single = SINGLE_RE.match(sig)
        if single:
            max_q = max(max_q, int(single.group(2)))
            continue
        cnot = CNOT_RE.match(sig)
        if cnot:
            max_q = max(max_q, int(cnot.group(1)), int(cnot.group(2)))
            continue
        raise ValueError(f"Unsupported signature: {sig}")
    return max_q + 1


def short_key(key: str) -> str:
    parts = key.split("__")
    if len(parts) < 5:
        return key
    return "/".join(parts[-3:])


def pairwise_jaccard(cluster_rows: list[dict[str, str]]) -> float:
    values = []
    sets = [set(row["sig_list"]) for row in cluster_rows]
    for a, b in itertools.combinations(sets, 2):
        values.append(len(a & b) / len(a | b))
    return sum(values) / len(values) if values else 1.0


def svg_header(width: int, height: int) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<rect x="0" y="0" width="100%" height="100%" fill="#fffdf8"/>',
    ]


def esc(text: str) -> str:
    return html.escape(text, quote=True)


def draw_svg(
    title: str,
    subtitle: str,
    sig_list: list[str],
    highlight_positions_set: set[int],
    output_path: Path,
) -> None:
    n_qubits = infer_num_qubits(sig_list)
    gate_w = 42
    dx = 58
    left = 86
    top = 86
    wire_gap = 40
    legend_h = 64
    right = 40
    bottom = 24
    width = left + right + max(1, len(sig_list)) * dx
    height = top + bottom + legend_h + max(1, n_qubits - 1) * wire_gap + 48

    y_for = lambda q: top + q * wire_gap

    core_fill = "#ffcc66"
    core_stroke = "#a85a00"
    non_fill = "#d9dde7"
    non_stroke = "#5a6472"
    wire_color = "#7a7f88"
    text_color = "#1f2430"

    parts = svg_header(width, height)
    parts.append(
        f'<text x="{left}" y="30" font-size="22" font-family="Helvetica,Arial,sans-serif" '
        f'font-weight="700" fill="{text_color}">{esc(title)}</text>'
    )
    parts.append(
        f'<text x="{left}" y="56" font-size="14" font-family="Helvetica,Arial,sans-serif" '
        f'fill="#4d5560">{esc(subtitle)}</text>'
    )

    for q in range(n_qubits):
        y = y_for(q)
        x1 = left - 32
        x2 = left + dx * (len(sig_list) - 1) + 24
        parts.append(
            f'<line x1="{x1}" y1="{y}" x2="{x2}" y2="{y}" stroke="{wire_color}" '
            f'stroke-width="2"/>'
        )
        parts.append(
            f'<text x="{left - 44}" y="{y + 5}" font-size="14" '
            f'font-family="Menlo,Consolas,monospace" fill="{text_color}">q{q}</text>'
        )

    for idx, sig in enumerate(sig_list):
        x = left + idx * dx
        highlighted = idx in highlight_positions_set
        fill = core_fill if highlighted else non_fill
        stroke = core_stroke if highlighted else non_stroke
        stroke_w = 2.8 if highlighted else 1.8

        if idx % 5 == 0:
            parts.append(
                f'<text x="{x - 5}" y="{top - 18}" font-size="11" '
                f'font-family="Menlo,Consolas,monospace" fill="#9096a0">{idx}</text>'
            )

        single = SINGLE_RE.match(sig)
        if single:
            gate, q = single.group(1), int(single.group(2))
            y = y_for(q)
            rect_x = x - gate_w / 2
            rect_y = y - 14
            parts.append(
                f'<rect x="{rect_x:.1f}" y="{rect_y:.1f}" width="{gate_w}" height="28" '
                f'rx="6" ry="6" fill="{fill}" stroke="{stroke}" stroke-width="{stroke_w}"/>'
            )
            parts.append(
                f'<text x="{x}" y="{y + 4}" text-anchor="middle" font-size="12" '
                f'font-family="Menlo,Consolas,monospace" font-weight="700" fill="{text_color}">{gate}</text>'
            )
            continue

        cnot = CNOT_RE.match(sig)
        if cnot:
            control = int(cnot.group(1))
            target = int(cnot.group(2))
            y1 = y_for(control)
            y2 = y_for(target)
            parts.append(
                f'<line x1="{x}" y1="{y1}" x2="{x}" y2="{y2}" stroke="{stroke}" '
                f'stroke-width="{stroke_w}"/>'
            )
            parts.append(
                f'<circle cx="{x}" cy="{y1}" r="5.4" fill="{stroke}" stroke="{stroke}" '
                f'stroke-width="1"/>'
            )
            parts.append(
                f'<circle cx="{x}" cy="{y2}" r="11" fill="{fill}" stroke="{stroke}" '
                f'stroke-width="{stroke_w}"/>'
            )
            parts.append(
                f'<line x1="{x}" y1="{y2 - 7}" x2="{x}" y2="{y2 + 7}" stroke="{stroke}" '
                f'stroke-width="2.2"/>'
            )
            parts.append(
                f'<line x1="{x - 7}" y1="{y2}" x2="{x + 7}" y2="{y2}" stroke="{stroke}" '
                f'stroke-width="2.2"/>'
            )
            continue

        raise ValueError(f"Unsupported signature: {sig}")

    legend_y = top + max(1, n_qubits - 1) * wire_gap + 48
    parts.append(
        f'<rect x="{left}" y="{legend_y}" width="24" height="18" rx="4" ry="4" '
        f'fill="{core_fill}" stroke="{core_stroke}" stroke-width="2"/>'
    )
    parts.append(
        f'<text x="{left + 34}" y="{legend_y + 13}" font-size="13" '
        f'font-family="Helvetica,Arial,sans-serif" fill="{text_color}">shared core</text>'
    )
    parts.append(
        f'<rect x="{left + 150}" y="{legend_y}" width="24" height="18" rx="4" ry="4" '
        f'fill="{non_fill}" stroke="{non_stroke}" stroke-width="2"/>'
    )
    parts.append(
        f'<text x="{left + 184}" y="{legend_y + 13}" font-size="13" '
        f'font-family="Helvetica,Arial,sans-serif" fill="{text_color}">representative-only context</text>'
    )

    parts.append("</svg>")
    output_path.write_text("\n".join(parts))


def write_notes(
    spec: FigureSpec,
    representative: dict[str, str],
    cluster_rows: list[dict[str, str]],
    common: Counter[str],
    output_path: Path,
) -> None:
    avg_jaccard = pairwise_jaccard(cluster_rows)
    core_size = sum(common.values())
    rep_size = len(representative["sig_list"])

    lines = [
        f"# {spec.title}: Representative Retained Circuit",
        "",
        f"- Representative snapshot: `{representative['episode_key']}`",
        f"- Stable cluster: `{spec.cluster_label}`",
        f"- Cluster size: `{len(cluster_rows)}`",
        f"- Representative retained gates: `{rep_size}`",
        f"- Shared multiset core size: `{core_size}`",
        f"- Average pairwise Jaccard overlap within cluster: `{avg_jaccard:.3f}`",
        f"- Notes: {spec.notes}",
        "",
        "## Cluster Members",
        "",
    ]

    for row in cluster_rows:
        lines.append(
            f"- `{short_key(row['episode_key'])}`: retained {row['retained_gate_count']} gates, "
            f"delta_error = {row['delta_error_mha']} mHa"
        )

    lines.extend(
        [
            "",
            "## Shared Core Signature Counts",
            "",
        ]
    )

    for sig, count in sorted(common.items()):
        lines.append(f"- `{sig}` x {count}")

    lines.extend(
        [
            "",
            "## Representative Retained Sequence",
            "",
            "```text",
            representative["retained_gates_with_angles"],
            "```",
            "",
            "## Figure",
            "",
            f"Generated SVG: `{output_path.with_suffix('.svg').name}`",
        ]
    )

    output_path.write_text("\n".join(lines) + "\n")


def main() -> None:
    for spec in SPECS:
        rows = load_rows(spec.summary_tsv)
        representative = rows[spec.representative_key]
        cluster_rows = [rows[key] for key in spec.cluster_keys]
        common = core_counter(cluster_rows)
        positions = highlighted_positions(representative["sig_list"], common)

        output_dir = spec.summary_tsv.parent
        svg_path = output_dir / "representative_core.svg"
        md_path = output_dir / "representative_core.md"

        subtitle = (
            f"highlighted gates = shared multiset core across {len(cluster_rows) - 1} other "
            f"stable snapshots"
        )
        draw_svg(spec.title, subtitle, representative["sig_list"], positions, svg_path)
        write_notes(spec, representative, cluster_rows, common, md_path)
        print(f"[written] {svg_path}")
        print(f"[written] {md_path}")


if __name__ == "__main__":
    main()
