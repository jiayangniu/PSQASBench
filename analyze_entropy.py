#!/usr/bin/env python3
"""
analyze_entropy.py — Per-qubit von Neumann entropy comparison for pruned circuits.

For each critical_structure_analysis/<bucket_dir>/ that contains a summary.tsv,
this script:
  1. Parses retained circuit structures from the 'retained_gates' column.
  2. Re-optimizes each circuit independently using the run's optimizer.
  3. Computes the per-qubit von Neumann entropy S(q) from the output state.
  4. Loads the exact ground state from the .npz Hamiltonian file and computes
     the exact per-qubit S(q).
  5. Reports qubit-by-qubit comparison and mean absolute error.

Usage:
    python analyze_entropy.py critical_structure_analysis/tier1_beh2_depth10/
    python analyze_entropy.py critical_structure_analysis/  # all subdirs
    python analyze_entropy.py dir1/ dir2/ dir3/

Options:
    --optimizer inherit|cobyla|rotosolve
    --n-restarts N
    --fidelity-threshold F   (for noting which circuits reached the ground state)
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

_AXIS_MAP = {"RX": 1, "RY": 2, "RZ": 3}
_GATE_ROT = re.compile(r"(RX|RY|RZ)\(q=(\d+)\)")
_GATE_CNOT = re.compile(r"CNOT\((\d+)->(\d+)\)")


# ── Gate string parsing (same as analyze_fidelity.py) ─────────────────────────

def parse_gate_string(s: str) -> list[GateSpec]:
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
    return specs


# ── Von Neumann entropy ────────────────────────────────────────────────────────

def single_qubit_entropies(state: np.ndarray, n_qubits: int) -> np.ndarray:
    """S(q) = -Tr(rho_q log2 rho_q) for each qubit q, in bits."""
    psi = np.asarray(state, dtype=np.complex128)
    psi = psi / np.linalg.norm(psi)
    tensor = psi.reshape((2,) * n_qubits)
    entropies = []
    for q in range(n_qubits):
        psi_q = np.moveaxis(tensor, q, 0).reshape(2, -1)
        rho_q = psi_q @ psi_q.conj().T
        evals = np.linalg.eigvalsh(rho_q)
        evals = np.clip(evals.real, 0.0, 1.0)
        mask = evals > 1e-14
        sq = float(-np.sum(evals[mask] * np.log2(evals[mask])))
        entropies.append(sq)
    return np.array(entropies)


def exact_ground_state_entropies(mol_path: Path, n_qubits: int) -> tuple[np.ndarray, float]:
    """Load NPZ, diagonalize, return (S(q) array, gap_mha)."""
    data = np.load(str(mol_path), allow_pickle=True)
    H = data["hamiltonian"].astype(np.complex128)
    energy_shift = float(data.get("energy_shift", 0.0))
    eigvals, eigvecs = np.linalg.eigh(H)
    gs = eigvecs[:, 0]
    gap_mha = (eigvals[1] - eigvals[0]) * 1000.0 if len(eigvals) > 1 else float("nan")
    sq_exact = single_qubit_entropies(gs, n_qubits)
    return sq_exact, gap_mha


# ── Circuit re-optimization (same pattern as analyze_fidelity.py) ──────────────

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
    n_rot = sum(1 for g in gates if g.gate_type == "rot")
    if optimizer != "rotosolve" or n_rot == 0:
        energy, opt_gates = _optimize_circuit(
            gates, n_qubits, H, energy_shift,
            optimizer=optimizer,
            maxiter=cobyla_maxiter,
            n_restarts=n_restarts,
            rotosolve_sweeps=rotosolve_sweeps,
            rng=rng,
        )
    else:
        best_energy = float("inf")
        best_opt_gates = gates
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


# ── TSV helpers ────────────────────────────────────────────────────────────────

def read_summary_tsv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [dict(row) for row in csv.DictReader(f, delimiter="\t")]


def _short_label(episode_key: str, fallback: str) -> str:
    m = re.search(r"ep(\d+)__snap(\d+)$", episode_key)
    if m:
        return f"ep{m.group(1)}_s{m.group(2)}"
    return fallback


# ── Per-directory analysis ─────────────────────────────────────────────────────

def analyze_dir(analysis_dir: Path, args: argparse.Namespace, rng: np.random.Generator) -> None:
    tsv_path = analysis_dir / "summary.tsv"
    if not tsv_path.exists():
        print(f"  [skip] no summary.tsv in {analysis_dir}")
        return

    rows = read_summary_tsv(tsv_path)
    if not rows:
        print(f"  [skip] empty summary.tsv in {analysis_dir}")
        return

    # Load run context (first valid row)
    ctx_cache: dict[str, RunContext] = {}
    mol_cache: dict[Path, tuple] = {}

    def get_ctx(run_dir_str: str) -> RunContext:
        if run_dir_str not in ctx_cache:
            ctx_cache[run_dir_str] = build_run_context(Path(run_dir_str))
        return ctx_cache[run_dir_str]

    def get_mol(mol_path: Path):
        if mol_path not in mol_cache:
            data = np.load(str(mol_path), allow_pickle=True)
            H = data["hamiltonian"].astype(np.complex128)
            energy_shift = float(data.get("energy_shift", 0.0))
            eigvals = np.linalg.eigvalsh(H)
            exact_energy = float(eigvals[0]) + energy_shift
            n_qubits = int(round(np.log2(H.shape[0])))
            mol_cache[mol_path] = (H, energy_shift, exact_energy, n_qubits, eigvals)
        return mol_cache[mol_path]

    # Determine molecule from first valid row
    mol_path = None
    for row in rows:
        run_dir_str = row.get("run_dir", "").strip()
        if run_dir_str:
            ctx = get_ctx(run_dir_str)
            mol_path = ctx.mol_path.resolve()
            break

    if mol_path is None:
        print(f"  [skip] could not resolve mol_path in {analysis_dir}")
        return

    H, energy_shift, exact_energy, n_qubits, eigvals = get_mol(mol_path)

    # Exact ground state S(q)
    data_npz = np.load(str(mol_path), allow_pickle=True)
    H_full = data_npz["hamiltonian"].astype(np.complex128)
    _, eigvecs_all = np.linalg.eigh(H_full)
    gs_exact = eigvecs_all[:, 0]
    sq_exact = single_qubit_entropies(gs_exact, n_qubits)
    gap_mha = (eigvals[1] - eigvals[0]) * 1000.0 if len(eigvals) > 1 else float("nan")
    degen_count = int(np.sum(np.abs(eigvals - eigvals[0]) <= 1e-6))

    print(f"  Molecule: {mol_path.name}")
    print(f"  n_qubits={n_qubits}  gap={gap_mha:.2f} mHa  GS_mult={degen_count}")
    print(f"  Exact S(q): {np.array2string(sq_exact, precision=4, separator=', ')}")

    print(f"  Processing {len(rows)} circuit(s) …")

    labels, sq_list, errors_mha, gate_counts = [], [], [], []

    for i, row in enumerate(rows):
        gate_str = row.get("retained_gates", "").strip()
        if not gate_str:
            continue
        run_dir_str = row.get("run_dir", "").strip()
        ctx = get_ctx(run_dir_str)

        gates = parse_gate_string(gate_str)
        if not gates:
            continue

        label = _short_label(row.get("episode_key", ""), fallback=f"circuit_{i}")

        if args.optimizer == "inherit":
            resolved_opt = row.get("resolved_optimizer", "cobyla").strip().lower()
            eff_maxiter = ctx.train_cobyla_maxiter or args.cobyla_maxiter if resolved_opt == "cobyla" else args.cobyla_maxiter
            eff_sweeps = ctx.train_rotosolve_sweeps or args.rotosolve_sweeps if resolved_opt == "rotosolve" else args.rotosolve_sweeps
        else:
            resolved_opt = args.optimizer
            eff_maxiter = args.cobyla_maxiter
            eff_sweeps = args.rotosolve_sweeps

        print(f"    [{i+1}/{len(rows)}] {label}: {len(gates)} gates [{resolved_opt}] … ", end="", flush=True)

        error_mha, vec = optimize_and_get_vec(
            gates, n_qubits, H, energy_shift, exact_energy,
            optimizer=resolved_opt,
            n_restarts=args.n_restarts,
            cobyla_maxiter=eff_maxiter,
            rotosolve_sweeps=eff_sweeps,
            rng=rng,
        )
        sq = single_qubit_entropies(vec, n_qubits)
        mae = float(np.mean(np.abs(sq - sq_exact)))
        print(f"err={error_mha:.4f} mHa  MAE_S={mae:.4f}")

        labels.append(label)
        sq_list.append(sq)
        errors_mha.append(error_mha)
        gate_counts.append(len(gates))

    if not sq_list:
        print(f"  [skip] no valid circuits in {analysis_dir}")
        return

    sq_arr = np.array(sq_list)  # shape (n_circuits, n_qubits)
    sq_mean = sq_arr.mean(axis=0)
    sq_std = sq_arr.std(axis=0)
    mae_per_circuit = [float(np.mean(np.abs(sq - sq_exact))) for sq in sq_list]
    max_dev_per_circuit = [float(np.max(np.abs(sq - sq_exact))) for sq in sq_list]

    # ── Write entropy_analysis.md ─────────────────────────────────────────────
    md: list[str] = [
        "# Per-Qubit Von Neumann Entropy Analysis",
        "",
        f"- molecule: `{mol_path.name}`",
        f"- n_qubits: `{n_qubits}`",
        f"- spectral gap: `{gap_mha:.4f} mHa`  |  GS multiplicity: `{degen_count}`",
        f"- circuits analyzed: `{len(labels)}`",
        f"- optimizer: `{args.optimizer}`  |  n_restarts: `{args.n_restarts}`",
        "",
        "## Exact Ground State S(q)",
        "",
        f"S(q) = {np.array2string(sq_exact, precision=4, separator=', ')}",
        f"  mean = {sq_exact.mean():.4f}  |  max = {sq_exact.max():.4f}  |  min = {sq_exact.min():.4f}",
        "",
        "## Per-Circuit Results",
        "",
        "| # | Label | Gates | Error (mHa) | S(q) circuit | MAE vs exact | max|ΔS| |",
        "|--:|-------|------:|------------:|:-------------|-------------:|------:|",
    ]
    for i, (lab, sq, err, ng, mae, mxd) in enumerate(
        zip(labels, sq_list, errors_mha, gate_counts, mae_per_circuit, max_dev_per_circuit)
    ):
        sq_str = "[" + ", ".join(f"{v:.3f}" for v in sq) + "]"
        md.append(f"| {i} | `{lab}` | {ng} | {err:.4f} | {sq_str} | {mae:.4f} | {mxd:.4f} |")

    # Summary row (mean across circuits)
    sq_mean_str = "[" + ", ".join(f"{v:.3f}" for v in sq_mean) + "]"
    sq_exact_str = "[" + ", ".join(f"{v:.3f}" for v in sq_exact) + "]"
    mean_mae = float(np.mean(mae_per_circuit))
    md += [
        f"| — | **mean** | — | {np.mean(errors_mha):.4f} | {sq_mean_str} | {mean_mae:.4f} | — |",
        f"| — | **exact** | — | 0.0000 | {sq_exact_str} | — | — |",
    ]

    # Per-qubit breakdown
    md += [
        "",
        "## Per-Qubit Comparison",
        "",
        "| Qubit | Exact S(q) | Mean circuit S(q) | Std | |ΔS| |",
        "|------:|----------:|------------------:|----:|----:|",
    ]
    for q in range(n_qubits):
        delta = abs(sq_mean[q] - sq_exact[q])
        md.append(
            f"| q{q} | {sq_exact[q]:.4f} | {sq_mean[q]:.4f} | {sq_std[q]:.4f} | {delta:.4f} |"
        )

    # Interpretation
    md += ["", "## Interpretation", ""]
    high_exact = [(q, sq_exact[q]) for q in range(n_qubits) if sq_exact[q] > 0.1]
    low_exact = [(q, sq_exact[q]) for q in range(n_qubits) if sq_exact[q] <= 0.05]

    if mean_mae < 0.05:
        md.append(
            f"> **Finding**: Mean |ΔS(q)| = {mean_mae:.4f} bits — pruned circuits "
            f"faithfully reproduce the exact ground-state entanglement profile."
        )
    elif mean_mae < 0.15:
        md.append(
            f"> **Finding**: Mean |ΔS(q)| = {mean_mae:.4f} bits — partial entanglement "
            f"match; pruned circuits capture the qualitative profile but with deviations."
        )
    else:
        md.append(
            f"> **Finding**: Mean |ΔS(q)| = {mean_mae:.4f} bits — significant mismatch "
            f"between pruned circuit S(q) and exact ground state. The circuit does not "
            f"encode the correct entanglement structure."
        )

    if high_exact:
        qs = ", ".join(f"q{q}({v:.3f})" for q, v in high_exact)
        md.append(f"> Highly entangled qubits in exact GS (S>0.1): {qs}")
    if low_exact:
        qs = ", ".join(f"q{q}" for q, _ in low_exact)
        md.append(f"> Near-product qubits in exact GS (S≤0.05): {qs}")

    if degen_count > 1:
        md.append(
            f"> **Degeneracy caveat**: GS multiplicity={degen_count} — exact S(q) depends "
            f"on the chosen eigenvector within the degenerate subspace and may not be unique."
        )

    md.append("")
    (analysis_dir / "entropy_analysis.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"    → mean MAE_S={mean_mae:.4f} | wrote entropy_analysis.md")


# ── Argument parsing ───────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Per-qubit von Neumann entropy comparison for pruned circuits."
    )
    p.add_argument("inputs", nargs="+", help="critical_structure_analysis dirs or parent dir")
    p.add_argument("--optimizer", choices=["cobyla", "rotosolve", "inherit"], default="inherit")
    p.add_argument("--n-restarts", type=int, default=10)
    p.add_argument("--cobyla-maxiter", type=int, default=2000)
    p.add_argument("--rotosolve-sweeps", type=int, default=2)
    p.add_argument("--rng-seed", type=int, default=42)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    rng = np.random.default_rng(args.rng_seed)

    targets: list[Path] = []
    for raw in args.inputs:
        path = Path(raw).expanduser().resolve()
        if not path.exists():
            print(f"[skip] not found: {path}")
            continue
        if path.is_dir() and (path / "summary.tsv").exists():
            targets.append(path)
        elif path.is_dir():
            found = sorted(p.parent for p in path.rglob("summary.tsv"))
            targets.extend(found)
        else:
            print(f"[skip] {path}")

    if not targets:
        print("No analysis directories found.")
        return 1

    for target in targets:
        print(f"\n[{target.name}]")
        analyze_dir(target, args, rng)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
