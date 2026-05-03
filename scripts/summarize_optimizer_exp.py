#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import re
from pathlib import Path


CONFIG_RE = re.compile(
    r"^bench_(?P<q>\d+)q_(?P<optim>[a-z0-9]+)_(?P<backend>cpu|gpu)_k(?P<k>\d+)\.cfg$"
)

OPTIM_ORDER = {"cobyla": 0, "rotosolve": 1, "adamspsa": 2, "psradam": 3}


def parse_run_meta(path: Path) -> dict[str, str]:
    data: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        if " = " not in raw:
            continue
        key, value = raw.split(" = ", 1)
        data[key.strip()] = value.strip()
    return data


def maybe_float(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def maybe_int(value: str | None) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except ValueError:
        return None


def collect_rows(results_root: Path) -> list[dict]:
    rows: list[dict] = []
    for meta_path in sorted(results_root.glob("crlqas/*/Optimizer_EXP/*/seed*/run_meta.txt")):
        meta = parse_run_meta(meta_path)
        cfg_name = meta.get("config_name", "")
        match = CONFIG_RE.match(cfg_name)
        if not match:
            continue

        row = {
            "mol": meta.get("mol", meta_path.parents[3].name),
            "config_name": cfg_name,
            "result_dir": meta.get("result_dir", str(meta_path.parent)),
            "status": meta.get("status", ""),
            "device": meta.get("device", ""),
            "seed": meta.get("seed", ""),
            "qubits": int(match.group("q")),
            "optimizer": match.group("optim"),
            "backend": match.group("backend"),
            "parallel_envs": int(match.group("k")),
            "wall_clock_sec": maybe_float(meta.get("wall_clock_sec")),
            "best_error_ha": maybe_float(meta.get("best_energy_error_ha")),
        }
        rows.append(row)
    return rows


def enrich_speedups(rows: list[dict]) -> None:
    baselines: dict[tuple[int, str, str], float] = {}
    cpu_baselines: dict[tuple[int, str], float] = {}
    for row in rows:
        wall = row.get("wall_clock_sec")
        if wall is None or wall <= 0:
            continue
        baselines[(row["qubits"], row["optimizer"], row["backend"])] = min(
            wall,
            baselines.get((row["qubits"], row["optimizer"], row["backend"]), math.inf),
        )
        if row["parallel_envs"] == 1 and row["backend"] == "cpu":
            cpu_baselines[(row["qubits"], row["optimizer"])] = wall

    k1_lookup: dict[tuple[int, str, str], float] = {}
    for row in rows:
        wall = row.get("wall_clock_sec")
        if wall is None or wall <= 0 or row["parallel_envs"] != 1:
            continue
        k1_lookup[(row["qubits"], row["optimizer"], row["backend"])] = wall

    for row in rows:
        wall = row.get("wall_clock_sec")
        row["speedup_vs_same_backend_k1"] = None
        row["speedup_vs_cpu_k1"] = None
        if wall is None or wall <= 0:
            continue
        k1 = k1_lookup.get((row["qubits"], row["optimizer"], row["backend"]))
        if k1 and wall > 0:
            row["speedup_vs_same_backend_k1"] = k1 / wall
        cpu_k1 = cpu_baselines.get((row["qubits"], row["optimizer"]))
        if cpu_k1 and wall > 0:
            row["speedup_vs_cpu_k1"] = cpu_k1 / wall


def write_tsv(rows: list[dict], path: Path) -> None:
    fieldnames = [
        "mol",
        "qubits",
        "optimizer",
        "backend",
        "parallel_envs",
        "device",
        "status",
        "wall_clock_sec",
        "speedup_vs_same_backend_k1",
        "speedup_vs_cpu_k1",
        "best_error_ha",
        "seed",
        "config_name",
        "result_dir",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def fmt_float(value) -> str:
    if value is None:
        return ""
    return f"{float(value):.4f}"


def write_md(rows: list[dict], path: Path) -> None:
    lines = [
        "# Optimizer Benchmark Summary",
        "",
        "| Mol | q | Optimizer | Backend | K | Status | Wall (s) | Speedup vs same-backend k=1 | Speedup vs cpu k=1 | Config |",
        "| --- | ---: | --- | --- | ---: | --- | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        lines.append(
            "| {mol} | {qubits} | {optimizer} | {backend} | {parallel_envs} | {status} | {wall} | {spd1} | {spdcpu} | `{cfg}` |".format(
                mol=row["mol"],
                qubits=row["qubits"],
                optimizer=row["optimizer"],
                backend=row["backend"],
                parallel_envs=row["parallel_envs"],
                status=row["status"],
                wall=fmt_float(row.get("wall_clock_sec")),
                spd1=fmt_float(row.get("speedup_vs_same_backend_k1")),
                spdcpu=fmt_float(row.get("speedup_vs_cpu_k1")),
                cfg=row["config_name"],
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize CRLQAS Optimizer_EXP timings.")
    parser.add_argument("--results-root", default="results", help="Benchmark results root")
    parser.add_argument("--out-tsv", default="optimizer_exp_summary.tsv", help="Output TSV path")
    parser.add_argument("--out-md", default="optimizer_exp_summary.md", help="Output Markdown path")
    args = parser.parse_args()

    results_root = Path(args.results_root).expanduser().resolve()
    rows = collect_rows(results_root)
    rows.sort(
        key=lambda r: (
            r["qubits"],
            OPTIM_ORDER.get(r["optimizer"], 999),
            r["backend"],
            r["parallel_envs"],
            r["config_name"],
        )
    )
    enrich_speedups(rows)

    write_tsv(rows, Path(args.out_tsv))
    write_md(rows, Path(args.out_md))
    print(f"Wrote {len(rows)} rows to {args.out_tsv} and {args.out_md}")


if __name__ == "__main__":
    main()
