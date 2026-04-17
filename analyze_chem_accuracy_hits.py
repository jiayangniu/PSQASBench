#!/usr/bin/env python3
"""Count which action reaches chemical accuracy in each episode.

For every episode in ``episode_traces.txt``, this script finds the first step
where ``energy_error <= accept_err`` and counts the corresponding action.
"""

from __future__ import annotations

import argparse
import ast
import configparser
import json
import re
from collections import Counter
from itertools import product
from pathlib import Path


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
    conf = {}
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Count which action first reaches chemical accuracy in each episode."
    )
    parser.add_argument(
        "path",
        help="Run directory containing episode_traces.txt, or the episode_traces.txt file itself.",
    )
    parser.add_argument(
        "--accept-err",
        type=float,
        default=None,
        help="Override the chemical-accuracy threshold. Default: read from run_meta.txt.",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=20,
        help="How many action rows to print.",
    )
    return parser.parse_args()


def resolve_paths(path_arg: str) -> tuple[Path, Path]:
    path = Path(path_arg).expanduser().resolve()
    if path.is_dir():
        run_dir = path
        traces_path = run_dir / "episode_traces.txt"
    else:
        traces_path = path
        run_dir = traces_path.parent
    if not traces_path.exists():
        raise FileNotFoundError(f"episode_traces.txt not found: {traces_path}")
    return run_dir, traces_path


def read_accept_err(run_dir: Path, override: float | None) -> float:
    if override is not None:
        return float(override)
    meta_path = run_dir / "run_meta.txt"
    if not meta_path.exists():
        raise FileNotFoundError(
            f"run_meta.txt not found in {run_dir}; pass --accept-err explicitly."
        )
    for line in meta_path.read_text(encoding="utf-8").splitlines():
        if line.startswith("accept_err ="):
            return float(line.split("=", 1)[1].strip())
    raise ValueError(f"accept_err not found in {meta_path}")


def read_action_decoder(run_dir: Path):
    cfg_path = run_dir / "config_used.cfg"
    if not cfg_path.exists():
        return None, None, None
    conf = read_cfg(cfg_path)
    num_qubits = int(conf["env"]["num_qubits"])
    connectivity = conf["env"].get("connectivity", "all")
    return num_qubits, connectivity, get_action_dict(num_qubits, connectivity)


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


def describe_action(action_id: int | None, action_dict: dict | None, num_qubits: int | None) -> str:
    if action_id is None:
        return "unknown"
    if action_dict is None or num_qubits is None:
        return f"action_id={action_id}"

    spec = action_dict.get(action_id)
    if spec is None:
        return f"action_id={action_id}"

    ctrl, offset, rot_qubit, rot_axis = spec
    if rot_qubit < num_qubits:
        axis_name = {1: "RX", 2: "RY", 3: "RZ"}.get(rot_axis, f"R{rot_axis}")
        return f"{axis_name}(q={rot_qubit})"

    targ = (ctrl + offset) % num_qubits
    return f"CNOT(ctrl={ctrl}, targ={targ}, offset={offset})"


def main() -> int:
    args = parse_args()
    run_dir, traces_path = resolve_paths(args.path)
    accept_err = read_accept_err(run_dir, args.accept_err)
    num_qubits, connectivity, action_dict = read_action_decoder(run_dir)
    episodes = parse_episode_traces(traces_path)

    hit_action_counter: Counter[int] = Counter()
    episodes_hit = 0

    for ep in episodes:
        actions = ep.get("actions", [])
        errors = ep.get("energy_errors", [])
        for idx, err in enumerate(errors):
            if float(err) <= accept_err:
                episodes_hit += 1
                if idx < len(actions):
                    action_item = actions[idx]
                    if isinstance(action_item, dict):
                        action_id = int(action_item["action"])
                    else:
                        action_id = int(action_item)
                    hit_action_counter[action_id] += 1
                break

    print(f"run_dir        : {run_dir}")
    print(f"traces_path    : {traces_path}")
    print(f"accept_err     : {accept_err}")
    if num_qubits is not None:
        print(f"num_qubits     : {num_qubits}")
        print(f"connectivity   : {connectivity}")
    print(f"episodes_total : {len(episodes)}")
    print(f"episodes_hit   : {episodes_hit}")
    print()

    print("count\taction_id\taction")
    for action_id, count in hit_action_counter.most_common(args.top):
        print(f"{count}\t{action_id}\t{describe_action(action_id, action_dict, num_qubits)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
