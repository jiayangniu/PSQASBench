"""
Base class for all QAS method runners in PSQASBench.
Each method implements run() and greedy_episode(), and inherits periodic_eval().
"""
import contextlib
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import torch

from . import VQE as vc


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

    # ── Shared batched optimizer helpers ──────────────────────────────────────
    # Each method uses one CUDA stream per env so all K independent GPU kernels
    # run concurrently.  No structure-based grouping: in RL training, K envs
    # almost never share the same circuit structure, so grouping only adds
    # hashing overhead without ever finding groups larger than 1.

    @staticmethod
    def _use_streams(env_list):
        """True when multiple CUDA-backed envs are present."""
        return (
            torch.cuda.is_available()
            and len(env_list) > 1
            and all(e._batched_vqe.device.type == "cuda" for e in env_list)
        )

    def _eval_env_energies_batched(self, envs):
        """Evaluate current circuit energies for K envs in parallel.

        One CUDA stream per env; results are collected after a single
        synchronize().  Falls back to env.get_energy() for phys-noise envs.

        Returns dict[k] = (energy, energy_noiseless) indexed by position in
        the input list.
        """
        noiseless = [(i, e) for i, e in enumerate(envs) if not e.phys_noise]
        use_streams = self._use_streams([e for _, e in noiseless])
        streams = (
            [torch.cuda.Stream() for _ in noiseless] if use_streams else
            [None] * len(noiseless)
        )

        futures = {}  # position -> raw GPU tensor
        for (i, env), stream in zip(noiseless, streams):
            st = env.state
            bvqe = env._batched_vqe
            bdev = bvqe.device
            rot_pos = (st[:, env.num_qubits: env.num_qubits + 3] == 1).nonzero(as_tuple=True)
            n_params = len(rot_pos[0])
            if n_params > 0:
                angles = np.asarray(st[:, env.num_qubits + 3:][rot_pos].cpu(), dtype=np.float32)
                angle_t = torch.tensor(angles.reshape(1, -1), dtype=torch.float32, device=bdev)
            else:
                angle_t = torch.empty((1, 0), dtype=torch.float32, device=bdev)
            ctx = torch.cuda.stream(stream) if stream else contextlib.nullcontext()
            with ctx:
                futures[i] = bvqe.eval_batch(st.to(bdev), angle_t)

        if use_streams:
            torch.cuda.synchronize()

        out = {}
        for i, env in noiseless:
            e_noiseless = float(futures[i].item())
            shot = vc._get_shot_noise(env.weights, env.n_shots)
            out[i] = (float(e_noiseless + shot), e_noiseless)

        for i, env in enumerate(envs):
            if i not in out:
                out[i] = env.get_energy()
        return out

    def _rotosolve_envs_batched(self, envs, opt_states=None):
        """Rotosolve for K envs with one CUDA stream per env per sweep.

        Within each sweep all eval_batch kernels are launched concurrently
        (one per stream), then a single synchronize() reads results before
        analytical angle updates.
        """
        if opt_states is None:
            opt_states = [env.state.clone() for env in envs]

        probe_vals = np.array([0.0, np.pi / 2.0, np.pi], dtype=np.float32)

        meta = []
        for k, env in enumerate(envs):
            st = opt_states[k]
            rot_pos = (st[:, env.num_qubits: env.num_qubits + 3] == 1).nonzero(as_tuple=True)
            n_params = len(rot_pos[0])
            if n_params == 0:
                continue
            angles = np.asarray(st[:, env.num_qubits + 3:][rot_pos].cpu(), dtype=np.float32)
            meta.append({
                "k": k, "env": env, "state": st,
                "rot_pos": rot_pos, "n": n_params,
                "angles": angles,
                "sweeps": int(getattr(env, "rotosolve_sweeps", 1)),
                "bdev": env._batched_vqe.device,
            })

        if not meta:
            return

        use_streams = self._use_streams([m["env"] for m in meta])
        streams = (
            [torch.cuda.Stream() for _ in meta] if use_streams else
            [None] * len(meta)
        )
        max_sweeps = max(m["sweeps"] for m in meta)

        for sweep_idx in range(max_sweeps):
            e_tensors = [None] * len(meta)

            # Phase 1: launch all kernels (skip envs that have finished their sweeps)
            for i, (m, stream) in enumerate(zip(meta, streams)):
                if sweep_idx >= m["sweeps"]:
                    continue
                n, angles, bdev = m["n"], m["angles"], m["bdev"]
                probe = np.tile(angles, (3 * n, 1))   # (3n, n)
                for p in range(n):
                    probe[3 * p: 3 * p + 3, p] = probe_vals
                ctx = torch.cuda.stream(stream) if stream else contextlib.nullcontext()
                with ctx:
                    angle_t = torch.tensor(probe, dtype=torch.float32, device=bdev)
                    e_tensors[i] = m["env"]._batched_vqe.eval_batch(m["state"].to(bdev), angle_t)

            # Phase 2: one barrier
            if use_streams:
                torch.cuda.synchronize()

            # Phase 3: analytical updates
            for i, m in enumerate(meta):
                if e_tensors[i] is None:
                    continue
                energies = e_tensors[i].cpu().numpy()
                angles = m["angles"]
                for p in range(m["n"]):
                    e0, epi2, epi = energies[3 * p], energies[3 * p + 1], energies[3 * p + 2]
                    A = (e0 - epi) / 2.0
                    C = (e0 + epi) / 2.0
                    B = epi2 - C
                    angles[p] = float(np.arctan2(-B, -A))

        for m in meta:
            th = envs[m["k"]].state[:, envs[m["k"]].num_qubits + 3:]
            th[m["rot_pos"]] = torch.tensor(m["angles"], dtype=torch.float32)

    def _psr_adam_envs_batched(self, envs, opt_states=None):
        """PSR + Adam for K envs with one CUDA stream per env per Adam step.

        Per Adam step: K probe matrices (2*n_params each) are evaluated
        concurrently, one GPU kernel per stream.  Exact gradients via PSR;
        vectorised Adam update per env.
        """
        if opt_states is None:
            opt_states = [env.state.clone() for env in envs]

        meta = []
        for k, env in enumerate(envs):
            st = opt_states[k]
            rot_pos = (st[:, env.num_qubits: env.num_qubits + 3] == 1).nonzero(as_tuple=True)
            n_params = len(rot_pos[0])
            if n_params == 0:
                continue
            raw = np.asarray(st[:, env.num_qubits + 3:][rot_pos].cpu(), dtype=np.float32)
            # Break θ≈0 symmetry: PSR gradient is zero when all angles=0 and H is real
            zm = np.abs(raw) < 1e-6
            raw[zm] = np.random.uniform(-0.3, 0.3, zm.sum())
            opts = getattr(env, "options", {})
            meta.append({
                "k": k, "env": env, "state": st,
                "rot_pos": rot_pos, "n": n_params,
                "angles": raw.copy(),
                "bdev": env._batched_vqe.device,
                "lr": float(opts.get("lr", 0.01)),
                "b1": float(opts.get("beta_1", 0.9)),
                "b2": float(opts.get("beta_2", 0.999)),
                "K_steps": max(1, int(env.global_iters)),
                "m": np.zeros(n_params, dtype=np.float32),
                "v": np.zeros(n_params, dtype=np.float32),
                "best_val": np.inf,
                "best_angles": raw.copy(),
            })

        if not meta:
            return

        use_streams = self._use_streams([m["env"] for m in meta])
        streams = (
            [torch.cuda.Stream() for _ in meta] if use_streams else
            [None] * len(meta)
        )

        # Initial energy for best-solution tracking
        init_t = [None] * len(meta)
        for i, (m, stream) in enumerate(zip(meta, streams)):
            ctx = torch.cuda.stream(stream) if stream else contextlib.nullcontext()
            with ctx:
                a_t = torch.tensor(m["angles"].reshape(1, -1), dtype=torch.float32, device=m["bdev"])
                init_t[i] = m["env"]._batched_vqe.eval_batch(m["state"].to(m["bdev"]), a_t)
        if use_streams:
            torch.cuda.synchronize()
        for i, m in enumerate(meta):
            m["best_val"] = float(init_t[i].item())

        K_steps = max(m["K_steps"] for m in meta)
        for step in range(K_steps):
            e_tensors = [None] * len(meta)

            # Phase 1: PSR probes (2*n per env) in parallel
            for i, (m, stream) in enumerate(zip(meta, streams)):
                if step >= m["K_steps"]:
                    continue
                n, angles, bdev = m["n"], m["angles"], m["bdev"]
                probe = np.repeat(angles.reshape(1, -1), 2 * n, axis=0)  # (2n, n)
                for p in range(n):
                    probe[2 * p,     p] += np.pi / 2
                    probe[2 * p + 1, p] -= np.pi / 2
                ctx = torch.cuda.stream(stream) if stream else contextlib.nullcontext()
                with ctx:
                    probe_t = torch.tensor(probe, dtype=torch.float32, device=bdev)
                    e_tensors[i] = m["env"]._batched_vqe.eval_batch(m["state"].to(bdev), probe_t)

            if use_streams:
                torch.cuda.synchronize()

            # Phase 2: Adam update
            for i, m in enumerate(meta):
                if e_tensors[i] is None:
                    continue
                s = step + 1
                energies = e_tensors[i].cpu().numpy()  # (2n,)
                grad = (energies[0::2] - energies[1::2]) / 2.0
                m["m"] = m["b1"] * m["m"] + (1 - m["b1"]) * grad
                m["v"] = m["b2"] * m["v"] + (1 - m["b2"]) * grad ** 2
                mhat = m["m"] / (1 - m["b1"] ** s)
                vhat = m["v"] / (1 - m["b2"] ** s)
                m["angles"] = m["angles"] - m["lr"] * mhat / (np.sqrt(vhat) + 1e-8)

        # Final eval — keep better of initial vs optimised
        final_t = [None] * len(meta)
        for i, (m, stream) in enumerate(zip(meta, streams)):
            ctx = torch.cuda.stream(stream) if stream else contextlib.nullcontext()
            with ctx:
                a_t = torch.tensor(m["angles"].reshape(1, -1), dtype=torch.float32, device=m["bdev"])
                final_t[i] = m["env"]._batched_vqe.eval_batch(m["state"].to(m["bdev"]), a_t)
        if use_streams:
            torch.cuda.synchronize()

        for i, m in enumerate(meta):
            best = m["angles"] if float(final_t[i].item()) < m["best_val"] else m["best_angles"]
            th = envs[m["k"]].state[:, envs[m["k"]].num_qubits + 3:]
            th[m["rot_pos"]] = torch.tensor(best, dtype=torch.float32)

    def _spsa_envs_batched(self, envs, opt_states=None, method="SPSA"):
        """SPSA / AdamSPSA for K envs with one CUDA stream per env per iteration.

        Per iteration: ±delta probes for all K envs are evaluated in parallel
        (one kernel per stream), followed by a current-val eval pass to track
        the best solution.
        """
        if opt_states is None:
            opt_states = [env.state.clone() for env in envs]

        use_adam = str(method).upper() == "ADAMSPSA"

        meta = []
        for k, env in enumerate(envs):
            st = opt_states[k]
            rot_pos = (st[:, env.num_qubits: env.num_qubits + 3] == 1).nonzero(as_tuple=True)
            n_params = len(rot_pos[0])
            if n_params == 0:
                continue
            angles = np.asarray(st[:, env.num_qubits + 3:][rot_pos].cpu(), dtype=np.float32)
            opts = getattr(env, "options", {})
            meta.append({
                "k": k, "env": env, "state": st,
                "rot_pos": rot_pos, "n": n_params,
                "angles": angles.reshape(1, -1).copy(),   # (1, n)
                "bdev": env._batched_vqe.device,
                "a": float(opts.get("a", 0.05)),
                "alpha": float(opts.get("alpha", 0.602)),
                "c": float(opts.get("c", 0.1)),
                "gamma": float(opts.get("gamma", 0.101)),
                "A": float(opts.get("lamda", max(10.0, env.global_iters * 0.1))),
                "b1": float(opts.get("beta_1", 0.9)),
                "b2": float(opts.get("beta_2", 0.999)),
                "iters": max(1, int(env.global_iters)),
                "m": np.zeros((1, n_params), dtype=np.float32),
                "v": np.zeros((1, n_params), dtype=np.float32),
                "best_val": np.inf,
                "best_angles": angles.reshape(1, -1).copy(),
            })

        if not meta:
            return

        use_streams = self._use_streams([m["env"] for m in meta])
        streams = (
            [torch.cuda.Stream() for _ in meta] if use_streams else
            [None] * len(meta)
        )

        # Initial best values
        init_t = [None] * len(meta)
        for i, (m, stream) in enumerate(zip(meta, streams)):
            ctx = torch.cuda.stream(stream) if stream else contextlib.nullcontext()
            with ctx:
                a_t = torch.tensor(m["angles"], dtype=torch.float32, device=m["bdev"])
                init_t[i] = m["env"]._batched_vqe.eval_batch(m["state"].to(m["bdev"]), a_t)
        if use_streams:
            torch.cuda.synchronize()
        for i, m in enumerate(meta):
            m["best_val"] = float(init_t[i].item())

        max_iters = max(m["iters"] for m in meta)
        for epoch in range(max_iters):
            # Phase 1: ±delta probes in parallel
            e_pm = [None] * len(meta)
            for i, (m, stream) in enumerate(zip(meta, streams)):
                if epoch >= m["iters"]:
                    continue
                ck = vc._spsa_ck(epoch, c=m["c"], gamma=m["gamma"])
                delta = np.random.choice([-1.0, 1.0], size=m["angles"].shape).astype(np.float32)
                m["_delta"] = delta
                m["_ck"] = ck
                probe = np.concatenate([m["angles"] + ck * delta,
                                        m["angles"] - ck * delta], axis=0)  # (2, n)
                ctx = torch.cuda.stream(stream) if stream else contextlib.nullcontext()
                with ctx:
                    probe_t = torch.tensor(probe, dtype=torch.float32, device=m["bdev"])
                    e_pm[i] = m["env"]._batched_vqe.eval_batch(m["state"].to(m["bdev"]), probe_t)
            if use_streams:
                torch.cuda.synchronize()

            # Phase 2: gradient + angle update
            for i, m in enumerate(meta):
                if epoch >= m["iters"] or e_pm[i] is None:
                    continue
                e_vals = e_pm[i].cpu().numpy()   # (2,)
                ak = vc._spsa_lr(epoch, a=m["a"], A=m["A"], alpha=m["alpha"])
                ck, delta = m["_ck"], m["_delta"]
                grad = ((e_vals[0] - e_vals[1]) / (2.0 * ck)) / delta   # (1, n)
                if use_adam:
                    m["m"] = m["b1"] * m["m"] + (1 - m["b1"]) * grad
                    m["v"] = m["b2"] * m["v"] + (1 - m["b2"]) * grad ** 2
                    mhat = m["m"] / (1 - m["b1"] ** (epoch + 1))
                    vhat = m["v"] / (1 - m["b2"] ** (epoch + 1))
                    m["angles"] = m["angles"] - ak * mhat / (np.sqrt(vhat) + 1e-8)
                else:
                    m["angles"] = m["angles"] - ak * grad

            # Phase 3: current-val evals in parallel (for best tracking)
            curr_t = [None] * len(meta)
            for i, (m, stream) in enumerate(zip(meta, streams)):
                if epoch >= m["iters"]:
                    continue
                ctx = torch.cuda.stream(stream) if stream else contextlib.nullcontext()
                with ctx:
                    a_t = torch.tensor(m["angles"], dtype=torch.float32, device=m["bdev"])
                    curr_t[i] = m["env"]._batched_vqe.eval_batch(m["state"].to(m["bdev"]), a_t)
            if use_streams:
                torch.cuda.synchronize()
            for i, m in enumerate(meta):
                if curr_t[i] is None:
                    continue
                val = float(curr_t[i].item())
                if val < m["best_val"]:
                    m["best_val"] = val
                    m["best_angles"] = m["angles"].copy()

        for m in meta:
            th = envs[m["k"]].state[:, envs[m["k"]].num_qubits + 3:]
            th[m["rot_pos"]] = torch.tensor(m["best_angles"].flatten(), dtype=torch.float32)
