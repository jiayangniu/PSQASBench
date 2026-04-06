"""
Method-agnostic evaluation utilities for PSQASBench.

Two rollout modes, both using the same standard result dict schema:

    greedy_fn(env) -> dict
        ε = 0, deterministic.  Used for SR / CNOT metrics.

    stochastic_fn(env, seed) -> dict
        Current training ε, seed-controlled.  Used for PCD.
        Different seeds → different exploration paths → meaningful diversity.

Standard rollout-result dict schema:
    energy_error : float   |E - E_exact| in Ha
    cnot_count   : int     CNOT gates in the generated circuit
    success      : int     1 if energy_error < accept_err, else 0
    steps        : int     number of env steps taken
    op_history   : list    gate sequence — used by pcd.compute_pcd
    n_qubits     : int     qubit count — used by pcd.compute_pcd
    final_state  : tensor  L × (N+6) × N state tensor — used by pcd.compute_pcd
"""

import copy
import numpy as np


# ── Rollout ────────────────────────────────────────────────────────────────────

def _save_env_eval_state(env) -> dict:
    """Capture optional curriculum-related state without assuming every env defines it."""
    saved = {}
    for attr in ("curriculum_dict", "curriculum", "done_threshold"):
        if hasattr(env, attr):
            saved[attr] = copy.deepcopy(getattr(env, attr))
    return saved


def _restore_env_eval_state(env, saved: dict) -> None:
    """Restore optional curriculum-related state and remove attrs that did not exist."""
    for attr in ("curriculum_dict", "curriculum", "done_threshold"):
        if attr in saved:
            setattr(env, attr, saved[attr])
        elif hasattr(env, attr) and attr == "curriculum":
            delattr(env, attr)

def greedy_rollout_k(greedy_fn, env, K: int) -> list[dict]:
    """
    Run K greedy episodes using the current policy.

    Curriculum state (curriculum_dict, curriculum, done_threshold) is saved
    before and fully restored after all K rollouts, so training is not affected.

    Args:
        greedy_fn : callable(env) -> dict
                    Provided by each Runner (e.g. CRLQASRunner._greedy_episode).
                    Must return the standard rollout-result dict (see module doc).
        env       : CircuitEnv — the *training* environment instance.
        K         : number of greedy rollouts.

    Returns:
        List of K rollout-result dicts.
    """
    # ── save curriculum state ─────────────────────────────────────────────────
    saved_state = _save_env_eval_state(env)

    results = []
    try:
        for _ in range(K):
            results.append(greedy_fn(env))
    finally:
        # ── restore curriculum state (always, even if greedy_fn raises) ───────
        _restore_env_eval_state(env, saved_state)

    return results


def stochastic_rollout_k(stochastic_fn, env, K: int, base_seed: int) -> list[dict]:
    """
    Run K stochastic episodes using the current training policy (ε unchanged).

    Each rollout uses a different seed (base_seed, base_seed+1, …, base_seed+K-1),
    so ε-greedy exploration produces K genuinely different circuits.
    Used exclusively for PCD computation.

    Curriculum state is saved before and restored after all K rollouts.

    Args:
        stochastic_fn : callable(env, seed) -> dict
                        Provided by each Runner (e.g. CRLQASRunner.stochastic_episode).
        env           : CircuitEnv — the *training* environment instance.
        K             : number of stochastic rollouts.
        base_seed     : seed for the first rollout; subsequent rollouts use base_seed+k.

    Returns:
        List of K rollout-result dicts (same schema as greedy_rollout_k).
    """
    saved_state = _save_env_eval_state(env)

    results = []
    try:
        for k in range(K):
            results.append(stochastic_fn(env, seed=base_seed + k))
    finally:
        _restore_env_eval_state(env, saved_state)

    return results


# ── Metric aggregation ─────────────────────────────────────────────────────────

def aggregate_metrics(rollout_results: list[dict], accept_err: float) -> dict:
    """
    Compute standard benchmark metrics from K rollout results.

    Args:
        rollout_results : list returned by greedy_rollout_k.
        accept_err      : chemical accuracy threshold in Ha (default 0.0016).

    Returns:
        dict with keys:
            sr_at_chem      float   Success rate (fraction reaching accept_err)
            cnot_at_chem    float   Median CNOT count among successful rollouts
                                    (NaN if no rollout succeeded)
            best_error_mha  float   Minimum energy error across all rollouts (mHa)
            mean_error_mha  float   Mean energy error (mHa)
            mean_cnots      float   Mean CNOT count across all rollouts
    """
    errors    = [r["energy_error"] for r in rollout_results]
    cnots     = [r["cnot_count"]   for r in rollout_results]
    successes = [r["success"]      for r in rollout_results]

    sr = float(np.mean(successes))

    succ_cnots = [cnots[i] for i, s in enumerate(successes) if s]
    cnot_at_chem = float(np.median(succ_cnots)) if succ_cnots else float("nan")

    return {
        "sr_at_chem":     sr,
        "cnot_at_chem":   cnot_at_chem,
        "best_error_mha": float(np.min(errors)) * 1000.0,
        "mean_error_mha": float(np.mean(errors)) * 1000.0,
        "mean_cnots":     float(np.mean(cnots)),
    }
