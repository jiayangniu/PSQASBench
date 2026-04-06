"""
Utility functions for PSQASBench RLQAS module.
Adapted from CRLQAS/utils.py and CRLQAS/utils_qas.py.
"""

import configparser
import json
import random
from itertools import product

import numpy as np
import torch
from qulacs import Observable


# ── Seed ─────────────────────────────────────────────────────────────────────

def set_seed(seed: int):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    if torch.cuda.device_count() > 0:
        torch.cuda.manual_seed_all(seed)


# ── Action dictionaries ───────────────────────────────────────────────────────

def dictionary_of_actions(num_qubits):
    """Maps integer action index → 4-tuple [ctrl, offset, rot_qubit, rot_axis]."""
    d = {}
    i = 0
    for c, x in product(range(num_qubits), range(1, num_qubits)):
        d[i] = [c, x, num_qubits, 0]
        i += 1
    for r, h in product(range(num_qubits), range(1, 4)):
        d[i] = [num_qubits, 0, r, h]
        i += 1
    return d


def dictionary_of_actions_linear(num_qubits):
    """Maps integer action index → 4-tuple [ctrl, offset, rot_qubit, rot_axis].
    Only nearest-neighbour CNOT gates (open chain, no wrap-around):
      ctrl → ctrl+1  (offset=1,           ctrl in 0..N-2)
      ctrl → ctrl-1  (offset=num_qubits-1, ctrl in 1..N-1)
    Rotation gates are unrestricted (all qubits × all axes).
    """
    d = {}
    i = 0
    for c in range(num_qubits - 1):          # forward edges: ctrl → ctrl+1
        d[i] = [c, 1, num_qubits, 0]
        i += 1
    for c in range(1, num_qubits):           # backward edges: ctrl → ctrl-1
        d[i] = [c, num_qubits - 1, num_qubits, 0]
        i += 1
    for r, h in product(range(num_qubits), range(1, 4)):
        d[i] = [num_qubits, 0, r, h]
        i += 1
    return d


def get_action_dict(num_qubits, connectivity='all'):
    """Factory: return action dictionary for given connectivity.
    connectivity='all'    → all-to-all CNOT (default)
    connectivity='linear' → 1D nearest-neighbor open chain
    """
    if connectivity == 'linear':
        return dictionary_of_actions_linear(num_qubits)
    return dictionary_of_actions(num_qubits)


def dict_of_actions_revert_q(num_qubits):
    """Same as above but with reversed qubit ordering."""
    d = {}
    i = 0
    for c, x in product(range(num_qubits - 1, -1, -1),
                         range(num_qubits - 1, 0, -1)):
        d[i] = [c, x, num_qubits, 0]
        i += 1
    for r, h in product(range(num_qubits - 1, -1, -1), range(1, 4)):
        d[i] = [num_qubits, 0, r, h]
        i += 1
    return d


def to_tuple4(x):
    """Normalise any sequence to a 4-int tuple."""
    if isinstance(x, np.ndarray):
        x = x.tolist()
    return tuple(int(v) for v in x)


# ── Config parser ─────────────────────────────────────────────────────────────

def get_config(cfg_path: str, verbose: bool = False) -> dict:
    """
    Parse a PSQASBench .cfg file and return a typed nested dict.

    Args:
        cfg_path : absolute path to the .cfg file
        verbose  : print parsed config if True
    """
    _FLOATS = {
        "learning_rate", "dropout", "alpha", "beta", "beta_incr",
        "shift_threshold_ball", "succes_switch", "tolearance_to_thresh",
        "memory_reset_threshold", "fake_min_energy", "_true_en",
        "n_shots", "err_mitig", "rand_halt",
        "a", "gamma", "c", "lamda", "beta_1", "beta_2",
        "refine_dropout", "delta_max", "refine_entropy_coef",
        "maxfev", "maxfev1", "maxfev2", "maxfev3",
    }
    _STRINGS = {
        "ham_type", "fn_type", "geometry", "method", "agent_type",
        "agent_class", "init_seed", "init_path", "init_thresh",
        "mapping", "optim_alg", "curriculum_type", "mol_file", "mol_data_dir",
        "connectivity", "refine_mode",
    }
    _LISTS = {
        "episodes", "neurons", "accept_err", "epsilon_decay", "epsilon_min",
        "final_gamma", "memory_clean", "update_target_net", "epsilon_restart",
        "thresholds", "switch_episodes", "refine_neurons",
    }

    config_dict = {}
    parser = configparser.ConfigParser()
    parser.read(cfg_path)

    for section in parser:
        if section == "DEFAULT":
            continue
        config_dict[section] = {}
        for key, val in parser.items(section):
            if key in _STRINGS:
                config_dict[section][key] = str(val)
            elif key in _LISTS:
                config_dict[section][key] = json.loads(val)
            elif key in _FLOATS:
                config_dict[section][key] = float(val)
            else:
                try:
                    config_dict[section][key] = int(val)
                except ValueError:
                    config_dict[section][key] = val

    if verbose:
        for sec, items in config_dict.items():
            print(f"[{sec}]")
            for k, v in items.items():
                print(f"  {k} = {v}")
    return config_dict


# ── Qulacs Observable builder ─────────────────────────────────────────────────

def map_theta(x) -> float:
    """Normalize angle to [-π, π]."""
    import math
    if hasattr(x, "item"):   # tensor scalar
        x = x.item()
    x = float(x) % (2 * math.pi)
    if x > math.pi:
        x -= 2 * math.pi
    return x


def build_observable(weights, pauli_strings, num_qubits) -> Observable:
    """Build a qulacs Observable from Pauli weights and string labels."""
    obs = Observable(num_qubits)
    for w, p in zip(weights, pauli_strings):
        obs.add_operator(float(w.real), str(p))
    return obs
