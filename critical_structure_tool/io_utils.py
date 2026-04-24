from __future__ import annotations

import ast
import configparser
import json
import re
from collections import Counter, defaultdict
from decimal import Decimal, ROUND_HALF_UP
from itertools import product
from pathlib import Path

from .types import RunContext, SnapshotRecord


REPO_ROOT = Path(__file__).resolve().parent.parent
MOL_DATA_DIR = REPO_ROOT / "mol_data"


def dictionary_of_actions(num_qubits: int) -> dict[int, list[int]]:
    action_dict = {}
    idx = 0
    for ctrl, offset in product(range(num_qubits), range(1, num_qubits)):
        action_dict[idx] = [ctrl, offset, num_qubits, 0]
        idx += 1
    for rot_qubit, rot_axis in product(range(num_qubits), range(1, 4)):
        action_dict[idx] = [num_qubits, 0, rot_qubit, rot_axis]
        idx += 1
    return action_dict


def dictionary_of_actions_linear(num_qubits: int) -> dict[int, list[int]]:
    action_dict = {}
    idx = 0
    for ctrl in range(num_qubits - 1):
        action_dict[idx] = [ctrl, 1, num_qubits, 0]
        idx += 1
    for ctrl in range(1, num_qubits):
        action_dict[idx] = [ctrl, num_qubits - 1, num_qubits, 0]
        idx += 1
    for rot_qubit, rot_axis in product(range(num_qubits), range(1, 4)):
        action_dict[idx] = [num_qubits, 0, rot_qubit, rot_axis]
        idx += 1
    return action_dict


def get_action_dict(num_qubits: int, connectivity: str = "all") -> dict[int, list[int]]:
    if connectivity == "linear":
        return dictionary_of_actions_linear(num_qubits)
    return dictionary_of_actions(num_qubits)


def read_cfg(cfg_path: Path) -> dict:
    parser = configparser.ConfigParser()
    parser.read(cfg_path)
    conf: dict[str, dict] = {}
    for section in parser.sections():
        conf[section] = {}
        for key, value in parser.items(section):
            if value.startswith("[") and value.endswith("]"):
                try:
                    conf[section][key] = json.loads(value)
                    continue
                except json.JSONDecodeError:
                    pass
            try:
                conf[section][key] = int(value)
                continue
            except ValueError:
                pass
            try:
                conf[section][key] = float(value)
                continue
            except ValueError:
                pass
            conf[section][key] = value
    return conf


def read_accept_err(run_dir: Path) -> float | None:
    meta_path = run_dir / "run_meta.txt"
    if not meta_path.exists():
        return None
    for line in meta_path.read_text(encoding="utf-8").splitlines():
        if line.startswith("accept_err_ha =") or line.startswith("accept_err ="):
            return float(line.split("=", 1)[1].strip())
    return None


def read_analysis_save_threshold(run_dir: Path) -> float | None:
    meta_path = run_dir / "run_meta.txt"
    if not meta_path.exists():
        return None
    for line in meta_path.read_text(encoding="utf-8").splitlines():
        if line.startswith("analysis_save_threshold_ha ="):
            value = line.split("=", 1)[1].strip()
            if value:
                return float(value)
    return None


def parse_episode_traces(traces_path: Path) -> list[dict]:
    episodes = []
    current: dict | None = None

    for raw_line in traces_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            if current is not None:
                episodes.append(current)
                current = None
            continue

        match = re.match(r"^\[episode\s+(\d+)\]$", line)
        if match:
            if current is not None:
                episodes.append(current)
            current = {"episode": int(match.group(1))}
            continue

        if current is None or "=" not in line:
            continue

        key, value = [part.strip() for part in line.split("=", 1)]
        normalized = (
            value.replace("null", "None")
            .replace("true", "True")
            .replace("false", "False")
        )
        current[key] = ast.literal_eval(normalized)

    if current is not None:
        episodes.append(current)

    return episodes


def get_trace_series(ep: dict, *candidate_keys: str) -> list:
    for key in candidate_keys:
        if key in ep:
            value = ep[key]
            if isinstance(value, list):
                return value
    return []


def get_analysis_snapshots(ep: dict) -> list[dict]:
    value = ep.get("analysis_snapshots", None)
    if isinstance(value, list):
        return [item for item in value if isinstance(item, dict)]
    legacy = ep.get("first_hit_snapshot", {})
    if isinstance(legacy, dict) and legacy:
        return [legacy]
    return []


def _last_token_from_gate_dicts(gate_dicts: list[dict]) -> str:
    """Derive a human-readable token for the last gate in a gates_direct list."""
    if not gate_dicts:
        return "unknown"
    d = gate_dicts[-1]
    gate_type = str(d.get("type", "")).lower()
    if gate_type == "cnot":
        return f"CNOT({d['ctrl']}->{d['targ']})"
    if gate_type == "rot":
        axis_name = {1: "RX", 2: "RY", 3: "RZ"}.get(int(d.get("axis", 0)), "R?")
        return f"{axis_name}(q={d['q']})"
    return "unknown"


def action_id_from_item(action_item) -> int:
    if isinstance(action_item, dict):
        return int(action_item["action"])
    return int(action_item)


def action_token(action_id: int, action_dict: dict[int, list[int]], num_qubits: int) -> str:
    spec = action_dict[action_id]
    ctrl, offset, rot_qubit, rot_axis = spec
    if rot_qubit < num_qubits:
        axis_name = {1: "RX", 2: "RY", 3: "RZ"}.get(rot_axis, f"R{rot_axis}")
        return f"{axis_name}(q={rot_qubit})"
    targ = (ctrl + offset) % num_qubits
    return f"CNOT({ctrl}->{targ})"


def discover_run_dirs(inputs: list[str]) -> list[Path]:
    run_dirs: set[Path] = set()
    for raw in inputs:
        path = Path(raw).expanduser().resolve()
        if path.is_file() and path.name == "episode_traces.txt":
            run_dirs.add(path.parent)
            continue
        if not path.exists():
            continue
        if path.is_dir() and (path / "episode_traces.txt").exists():
            run_dirs.add(path)
            continue
        if path.is_dir():
            for traces_path in path.rglob("episode_traces.txt"):
                run_dirs.add(traces_path.parent.resolve())
    return sorted(run_dirs)


def build_run_context(run_dir: Path) -> RunContext:
    cfg_path = run_dir / "config_used.cfg"
    conf = read_cfg(cfg_path)
    num_qubits = int(conf["env"]["num_qubits"])
    connectivity = str(conf["env"].get("connectivity", "all"))
    action_dict = get_action_dict(num_qubits, connectivity)
    mol_file = conf["problem"]["mol_file"]
    mol_path = MOL_DATA_DIR / mol_file
    non_local = conf.get("non_local_opt", {})
    train_optimizer = str(non_local.get("optim_alg", "COBYLA")).lower()
    train_cobyla_maxiter = None
    if "global_iters" in non_local:
        try:
            train_cobyla_maxiter = int(non_local["global_iters"])
        except (TypeError, ValueError):
            train_cobyla_maxiter = None
    train_rotosolve_sweeps = None
    if "rotosolve_sweeps" in non_local:
        try:
            train_rotosolve_sweeps = int(non_local["rotosolve_sweeps"])
        except (TypeError, ValueError):
            train_rotosolve_sweeps = None

    method = molecule = exp_group = config_name = "unknown"
    seed_name = run_dir.name
    parts = run_dir.parts
    if "results" in parts:
        idx = parts.index("results")
        rel = parts[idx + 1 :]
        if len(rel) >= 5:
            method, molecule, exp_group, config_name, seed_name = rel[:5]
        elif len(rel) >= 4:
            # Support direct-gate method layouts such as:
            #   results/qdarts/<molecule>/<config>/seed11111
            method, molecule, config_name, seed_name = rel[:4]
            exp_group = config_name

    return RunContext(
        run_dir=run_dir,
        method=method,
        molecule=molecule,
        exp_group=exp_group,
        config_name=config_name,
        seed_name=seed_name,
        accept_err_ha=read_accept_err(run_dir),
        analysis_save_threshold_ha=read_analysis_save_threshold(run_dir),
        num_qubits=num_qubits,
        connectivity=connectivity,
        action_dict=action_dict,
        mol_path=mol_path,
        train_optimizer=train_optimizer,
        train_cobyla_maxiter=train_cobyla_maxiter,
        train_rotosolve_sweeps=train_rotosolve_sweeps,
    )


def bucket_decimals(bucket_width_mha: float) -> int:
    text = format(bucket_width_mha, "f").rstrip("0").rstrip(".")
    if "." not in text:
        return 0
    return len(text.split(".", 1)[1])


def bucket_label(error_mha: float, bucket_width_mha: float) -> str:
    width = Decimal(str(bucket_width_mha))
    if width <= 0:
        raise ValueError("bucket_width_mha must be positive.")
    decimals = bucket_decimals(bucket_width_mha)
    quantized = (
        Decimal(str(error_mha)) / width
    ).quantize(Decimal("1"), rounding=ROUND_HALF_UP) * width
    return f"{quantized:.{decimals}f}"


def display_error_label(error_mha: float) -> str:
    if error_mha == 0.0:
        return "0"
    return f"{error_mha:.2g}"

def collect_snapshot_records(
    run_dirs: list[Path],
    target_error_mha: float | None,
    bucket_width_mha: float,
) -> list[SnapshotRecord]:
    records: list[SnapshotRecord] = []
    seen_keys: set[tuple] = set()  # (run_dir, episode_index, snapshot_index)

    for run_dir in run_dirs:
        ctx = build_run_context(run_dir)
        requested_target_error_ha = None if target_error_mha is None else target_error_mha / 1000.0
        episodes = parse_episode_traces(run_dir / "episode_traces.txt")
        total_eps = len(episodes)

        for ep in episodes:
            actions = ep.get("actions", [])
            errors = get_trace_series(ep, "energy_errors_ha", "energy_errors")
            action_ids = [action_id_from_item(item) for item in actions]
            snapshots = get_analysis_snapshots(ep)

            if snapshots:
                effective_target_error_ha = requested_target_error_ha
                if effective_target_error_ha is None:
                    # For modern runs with saved analysis snapshots, the default
                    # behavior is to analyze all snapshot events saved under the
                    # run's analysis_save_threshold.  Falling back to accept_err
                    # here would wrongly discard valid snapshots whenever
                    # analysis_save_threshold > accept_err.
                    effective_target_error_ha = (
                        ctx.analysis_save_threshold_ha
                        if ctx.analysis_save_threshold_ha is not None
                        else ctx.accept_err_ha
                    )
                for snap_idx, snapshot in enumerate(snapshots):
                    try:
                        step_idx = int(snapshot["step"])
                    except (KeyError, TypeError, ValueError):
                        continue

                    # Check for gates_direct (TFQAS/DARTS format) inside the snapshot.
                    gates_direct_raw = snapshot.get("gates_direct")
                    has_gates_direct = isinstance(gates_direct_raw, list) and len(gates_direct_raw) > 0

                    dedup_key = (run_dir, int(ep["episode"]), snap_idx)
                    if dedup_key in seen_keys:
                        continue
                    seen_keys.add(dedup_key)

                    if has_gates_direct:
                        # Direct-gate method: step_idx indexes into errors only.
                        if step_idx < 0 or step_idx >= len(errors):
                            continue
                        event_err_ha = float(errors[step_idx])
                        if effective_target_error_ha is not None and event_err_ha > effective_target_error_ha:
                            continue
                        event_err_mha = event_err_ha * 1000.0
                        records.append(
                            SnapshotRecord(
                                run=ctx,
                                episode_index=int(ep["episode"]),
                                total_episodes=total_eps,
                                snapshot_index=snap_idx,
                                event_step=step_idx + 1,
                                event_error_ha=event_err_ha,
                                event_error_mha=event_err_mha,
                                bucket=bucket_label(event_err_mha, bucket_width_mha),
                                action_ids_prefix=[],
                                last_action_id=-1,
                                last_action_token=_last_token_from_gate_dicts(gates_direct_raw),
                                snapshot=snapshot,
                                gates_direct=list(gates_direct_raw),
                            )
                        )
                    else:
                        # RL method: step_idx indexes into both action_ids and errors.
                        if step_idx < 0 or step_idx >= len(action_ids) or step_idx >= len(errors):
                            continue
                        event_err_ha = float(errors[step_idx])
                        if effective_target_error_ha is not None and event_err_ha > effective_target_error_ha:
                            continue
                        prefix_ids = action_ids[: step_idx + 1]
                        if not prefix_ids:
                            continue
                        last_action_id = prefix_ids[-1]
                        event_err_mha = event_err_ha * 1000.0
                        records.append(
                            SnapshotRecord(
                                run=ctx,
                                episode_index=int(ep["episode"]),
                                total_episodes=total_eps,
                                snapshot_index=snap_idx,
                                event_step=step_idx + 1,
                                event_error_ha=event_err_ha,
                                event_error_mha=event_err_mha,
                                bucket=bucket_label(event_err_mha, bucket_width_mha),
                                action_ids_prefix=prefix_ids,
                                last_action_id=last_action_id,
                                last_action_token=action_token(last_action_id, ctx.action_dict, ctx.num_qubits),
                                snapshot=snapshot,
                                gates_direct=None,
                            )
                        )
                continue

            effective_target_error_ha = requested_target_error_ha
            if effective_target_error_ha is None:
                effective_target_error_ha = ctx.accept_err_ha

            if effective_target_error_ha is None:
                raise ValueError(
                    f"No target error provided, no analysis snapshots found, and no accept_err found in run_meta.txt for {run_dir}"
                )

            hit_idx = None
            hit_err_ha = None
            for idx, err in enumerate(errors):
                if float(err) <= effective_target_error_ha:
                    hit_idx = idx
                    hit_err_ha = float(err)
                    break

            if hit_idx is None or hit_err_ha is None or hit_idx >= len(action_ids):
                continue

            hit_err_mha = hit_err_ha * 1000.0
            dedup_key = (run_dir, int(ep["episode"]), 0)
            if dedup_key in seen_keys:
                continue
            seen_keys.add(dedup_key)
            prefix_ids = action_ids[: hit_idx + 1]
            last_action_id = prefix_ids[-1]
            records.append(
                SnapshotRecord(
                    run=ctx,
                    episode_index=int(ep["episode"]),
                    total_episodes=total_eps,
                    snapshot_index=0,
                    event_step=hit_idx + 1,
                    event_error_ha=hit_err_ha,
                    event_error_mha=hit_err_mha,
                    bucket=bucket_label(hit_err_mha, bucket_width_mha),
                    action_ids_prefix=prefix_ids,
                    last_action_id=last_action_id,
                    last_action_token=action_token(last_action_id, ctx.action_dict, ctx.num_qubits),
                    snapshot=(snapshots[0] if snapshots else {}),
                )
            )
    return records


def summarize_first_hit_distribution(records: list[SnapshotRecord]) -> list[dict]:
    grouped: dict[str, list[SnapshotRecord]] = defaultdict(list)
    for rec in records:
        grouped[display_error_label(rec.event_error_mha)].append(rec)

    rows = []
    for label, recs in sorted(grouped.items(), key=lambda item: min(r.event_error_mha for r in item[1])):
        rows.append(
            {
                "display_value": label,
                "count": len(recs),
                "seed_count": len({str(r.run.run_dir) for r in recs}),
                "min_first_hit_error_mha": f"{min(r.event_error_mha for r in recs):.10f}",
                "max_first_hit_error_mha": f"{max(r.event_error_mha for r in recs):.10f}",
            }
        )
    return rows


def summarize_buckets(records: list[SnapshotRecord]) -> list[dict]:
    bucket_map: dict[str, list[SnapshotRecord]] = defaultdict(list)
    for rec in records:
        bucket_map[rec.bucket].append(rec)

    rows = []
    for bucket in sorted(bucket_map, key=lambda x: float(x)):
        recs = bucket_map[bucket]
        seed_count = len({str(rec.run.run_dir) for rec in recs})
        mean_hit_step = sum(rec.event_step for rec in recs) / len(recs)
        mean_episode = sum(rec.episode_index for rec in recs) / len(recs)
        rows.append(
            {
                "bucket": bucket,
                "count": len(recs),
                "seed_count": seed_count,
                "mean_hit_step": mean_hit_step,
                "mean_episode": mean_episode,
            }
        )
    return rows


def choose_bucket_interactive(
    bucket_rows: list[dict], requested: str | None, bucket_width_mha: float
) -> str:
    if not bucket_rows:
        raise ValueError("No snapshot events reached the requested target error threshold.")
    bucket_values = {row["bucket"] for row in bucket_rows}
    if requested is not None:
        normalized = bucket_label(float(requested), bucket_width_mha)
        if normalized not in bucket_values:
            raise ValueError(f"Bucket {normalized} not found in available buckets: {sorted(bucket_values)}")
        return normalized

    print("\nAvailable buckets (mHa):")
    print("bucket\tcount\tseed_count\tmean_hit_step\tmean_episode")
    for row in bucket_rows:
        print(
            f"{row['bucket']}\t{row['count']}\t{row['seed_count']}\t"
            f"{row['mean_hit_step']:.2f}\t{row['mean_episode']:.2f}"
        )

    try:
        answer = input("\nSelect the bucket to analyze (mHa): ").strip()
    except EOFError:
        answer = ""

    if not answer:
        return bucket_rows[0]["bucket"]
    normalized = bucket_label(float(answer), bucket_width_mha)
    if normalized not in bucket_values:
        raise ValueError(f"Bucket {normalized} not found in available buckets: {sorted(bucket_values)}")
    return normalized


def compute_anchor_action_counts(records: list[SnapshotRecord]) -> Counter[str]:
    return Counter(rec.last_action_token for rec in records)


def sample_representative_episodes(
    records: list[SnapshotRecord],
    n_select: int,
    late_fraction: float,
    episode_range: tuple[int, int] | None,
    rng,
) -> list[SnapshotRecord]:
    by_run: dict[Path, list[SnapshotRecord]] = defaultdict(list)
    for rec in records:
        by_run[rec.run.run_dir].append(rec)

    prepared: dict[Path, list[SnapshotRecord]] = {}
    for run_dir, recs in by_run.items():
        recs_sorted = sorted(recs, key=lambda r: (r.episode_index, r.snapshot_index))
        if episode_range is not None:
            start_ep, end_ep = episode_range
            pool = [r for r in recs_sorted if start_ep <= r.episode_index <= end_ep]
        else:
            min_episode = int(recs_sorted[-1].total_episodes * (1.0 - late_fraction))
            late_recs = [r for r in recs_sorted if r.episode_index >= min_episode]
            pool = late_recs if late_recs else recs_sorted
        pool = list(pool)
        rng.shuffle(pool)
        prepared[run_dir] = pool

    selected: list[SnapshotRecord] = []
    while len(selected) < n_select:
        progressed = False
        for run_dir in sorted(prepared):
            pool = prepared[run_dir]
            if not pool:
                continue
            selected.append(pool.pop(0))
            progressed = True
            if len(selected) >= n_select:
                break
        if not progressed:
            break
    return selected
