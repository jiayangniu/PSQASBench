"""
Base class for all QAS method runners in PSQASBench.
Each method implements run() and greedy_episode(), and inherits periodic_eval().
"""
from abc import ABC, abstractmethod
from pathlib import Path
import numpy as np


class BaseRunner(ABC):
    """Abstract base for all benchmark runners."""

    def __init__(self, config, mol_path: Path, result_dir: Path, seed: int):
        self.config     = config
        self.mol_path   = mol_path
        self.result_dir = result_dir
        self.seed       = seed

        result_dir.mkdir(parents=True, exist_ok=True)
        self._load_molecule()

    def _load_molecule(self):
        """Load Hamiltonian and exact ground state energy from .npz."""
        data = np.load(self.mol_path, allow_pickle=True)
        self.hamiltonian  = data["hamiltonian"]
        self.weights      = data["weights"]
        self.energy_shift = float(data.get("energy_shift", 0.0))
        self.exact_energy = float(np.min(data["eigvals"])) + self.energy_shift

    # ── Abstract interface ─────────────────────────────────────────────────────

    @abstractmethod
    def run(self) -> dict:
        """
        Execute the full QAS experiment (training + optional eval).

        Returns a standardised result dict:
            best_energy      float
            energy_error     float   |best_energy - exact_energy| in Ha
            cnot_count       int
            circuit_depth    int
            nfev             int     total VQE function evaluations (-1 if unknown)
            success          int     1 if energy_error < accept_err
            energy_history   list    energy per training episode
            cnot_history     list    CNOT count per training episode
            seed             int
        """
        raise NotImplementedError

    @abstractmethod
    def greedy_episode(self, env) -> dict:
        """
        Run a single deterministic (ε=0) rollout with the current policy.
        Used for SR / CNOT metrics.

        Implementations must:
          - Temporarily disable exploration (ε=0 for DQN, deterministic
            sampling for policy-gradient methods).
          - Call env.step(..., train_flag=False).
          - Restore all agent state before returning.

        Returns the standard rollout-result dict:
            energy_error  float
            cnot_count    int
            success       int
            steps         int
            op_history    list
            n_qubits      int
            final_state   tensor  L × (N+6) × N
        """
        raise NotImplementedError

    @abstractmethod
    def stochastic_episode(self, env, seed: int) -> dict:
        """
        Run a single stochastic rollout with the current training policy (ε unchanged).
        Used exclusively for PCD computation.

        Implementations must:
          - Set np/torch seed to `seed` before acting (for reproducibility).
          - Keep current exploration rate (do NOT set ε=0).
          - Call env.step(..., train_flag=False).
          - Restore all agent state before returning.

        Returns the same standard rollout-result dict schema as greedy_episode.
        """
        raise NotImplementedError

    # ── Shared periodic evaluation (called from training loops) ───────────────

    def periodic_eval(self, ep: int, env, K: int = 20) -> dict:
        """
        Evaluate the current policy periodically during training.

        - Greedy rollouts (ε=0) × K  →  SR@chem, CNOT@chem, best_error_mha
        - Stochastic rollouts (current ε, K different seeds) × K  →  D_struct, D_func

        Curriculum state of `env` is fully restored after both rollout sets.

        Args:
            ep  : current training episode number.
            env : CircuitEnv training instance.
            K   : rollouts per mode (default 20).

        Returns:
            Merged dict of aggregate_metrics + PCD results.
        """
        import sys, pathlib
        sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
        from metrics import (greedy_rollout_k, stochastic_rollout_k,
                             aggregate_metrics, compute_pcd)

        accept_err = self.config["env"]["accept_err"]

        # SR / CNOT metrics — deterministic greedy policy
        greedy_rollouts = greedy_rollout_k(self.greedy_episode, env, K)
        metrics = aggregate_metrics(greedy_rollouts, accept_err)

        # PCD — stochastic policy, K different seeds
        pcd_base_seed = ep  # different seed set at every eval checkpoint
        stoch_rollouts = stochastic_rollout_k(
            self.stochastic_episode, env, K, base_seed=pcd_base_seed
        )
        pcd = compute_pcd(
            stoch_rollouts,
            hamiltonian  = self.hamiltonian,
            weights      = self.weights,
            energy_shift = self.energy_shift,
            reoptimize   = False,
        )
        return {**metrics, **pcd, "eval_episode": ep}

    # ── Persistence ────────────────────────────────────────────────────────────

    def save_result(self, result: dict):
        """Save result dict as .npz in result_dir."""
        out_path = self.result_dir / f"result_seed{self.seed}.npz"
        np.savez(out_path, **{k: np.array(v) for k, v in result.items()})
        return out_path

    # ── Shared training dispatch ─────────────────────────────────────────────

    def train_with_parallel_entry(
        self,
        env,
        agent,
        episodes: int,
        train_single_fn,
        make_env_fn=None,
        train_parallel_fn=None,
        require_batch_divisible: bool = False,
    ):
        """Unified entry for single-env vs parallel-env training.

        All method runners should call this helper in run().
        If `general.num_parallel_envs > 1` and `train_parallel_fn` is provided,
        this helper creates K env instances via `make_env_fn` and runs the
        parallel trainer. Otherwise it falls back to the single-env trainer.
        """
        general = self.config.get("general", {})
        K = int(general.get("num_parallel_envs", 1))

        if K > 1 and train_parallel_fn is not None and make_env_fn is not None:
            if require_batch_divisible:
                batch_size = int(getattr(agent, "batch_size", self.config.get("agent", {}).get("batch_size", 0)))
                if batch_size <= 0:
                    raise ValueError("Agent must define positive batch_size for parallel training.")
                if batch_size % K != 0:
                    raise ValueError(
                        f"batch_size ({batch_size}) must be divisible by num_parallel_envs ({K})"
                    )
            envs = [make_env_fn() for _ in range(K)]
            return train_parallel_fn(envs, agent, episodes)

        return train_single_fn(env, agent, episodes)
