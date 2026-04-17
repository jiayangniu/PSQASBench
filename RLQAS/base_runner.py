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
from .utils import count_rotation_gates


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

        K stochastic rollouts (current training ε, K different seeds) are used
        for ALL metrics: SR@chem, CNOT@chem, best_error_mha, D_struct, D_func.

        Using stochastic rollouts makes SR a real probability (not a binary 0/1
        from K identical deterministic rollouts), and eliminates the redundant
        greedy rollout set.

        Curriculum state of `env` is fully restored after all rollouts.

        Args:
            ep  : current training episode number.
            env : CircuitEnv training instance.
            K   : number of stochastic rollouts (default 20).

        Returns:
            Merged dict of aggregate_metrics + PCD results.
        """
        import sys, pathlib
        sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
        from metrics import (stochastic_rollout_k, aggregate_metrics, compute_pcd)

        accept_err = self.config["env"]["accept_err"]

        # Single stochastic rollout set — serves both SR/CNOT and PCD
        pcd_base_seed = ep  # different seed set at every eval checkpoint
        stoch_rollouts = stochastic_rollout_k(
            self.stochastic_episode, env, K, base_seed=pcd_base_seed
        )
        metrics = aggregate_metrics(stoch_rollouts, accept_err)
        best_rollout = min(stoch_rollouts, key=lambda r: r["energy_error"])

        compute_pcd_flag = bool(int(self.config.get("general", {}).get("compute_pcd", 1)))
        if compute_pcd_flag:
            pcd = compute_pcd(
                stoch_rollouts,
                hamiltonian  = self.hamiltonian,
                weights      = self.weights,
                energy_shift = self.energy_shift,
                reoptimize   = False,
            )
        else:
            pcd = {
                "D_struct": float("nan"),
                "D_func": float("nan"),
                "n_pairs": 0,
            }
        best_rollout_info = {
            "best_rollout_energy": float(best_rollout.get("energy", float("nan"))),
            "best_rollout_energy_error": float(best_rollout["energy_error"]),
            "best_rollout_depth": int(best_rollout["steps"]),
            "best_rollout_cnot_count": int(best_rollout["cnot_count"]),
            "best_rollout_rotation_count": int(
                best_rollout.get(
                    "rotation_count",
                    count_rotation_gates(best_rollout.get("op_history", [])),
                )
            ),
            "best_rollout_op_history": list(best_rollout.get("op_history", [])),
        }
        return {**metrics, **pcd, **best_rollout_info, "eval_episode": ep}

    # ── Persistence ────────────────────────────────────────────────────────────

    def save_result(self, result: dict):
        """Save result dict as .npz in result_dir."""
        out_path = self.result_dir / f"result_seed{self.seed}.npz"
        np.savez(out_path, **{k: np.array(v) for k, v in result.items()})
        return out_path

    # ── Shared trace helpers ──────────────────────────────────────────────────

    @staticmethod
    def _capture_first_hit_snapshot(env) -> dict:
        """Capture the minimum parameter snapshot needed to reconstruct first-hit.

        The snapshot stores only:
          - the hit step index
          - optimized rotation parameters at that step
          - which action-prefix step each parameter belongs to

        All other information (energies, errors, threshold, actions) is already
        available from the surrounding trace / run metadata.
        """
        state = env.state.detach().cpu()
        gate_params = []
        param_step_indices = []

        for step_idx, rec in enumerate(env.op_history):
            if rec.get("type") != "rot":
                continue
            layer = int(rec["layer"])
            q = int(rec["q"])
            axis = int(rec["axis"])
            angle_plane = env.num_qubits + 3 + axis - 1
            gate_params.append(float(state[layer][angle_plane][q].item()))
            param_step_indices.append(step_idx)

        return {
            "step": int(env.step_counter),
            "gate_params": gate_params,
            "param_step_indices": param_step_indices,
        }

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
            bdev = env._batched_vqe.device
            meta.append({
                "k": k, "env": env, "state": st,
                "state_gpu": st.to(bdev),          # cached once; avoids repeated CPU→GPU copies
                "rot_pos": rot_pos, "n": n_params,
                "angles": raw.copy(),
                "bdev": bdev,
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
                init_t[i] = m["env"]._batched_vqe.eval_batch(m["state_gpu"], a_t)
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
                    e_tensors[i] = m["env"]._batched_vqe.eval_batch(m["state_gpu"], probe_t)

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
                final_t[i] = m["env"]._batched_vqe.eval_batch(m["state_gpu"], a_t)
        if use_streams:
            torch.cuda.synchronize()

        for i, m in enumerate(meta):
            best = m["angles"] if float(final_t[i].item()) < m["best_val"] else m["best_angles"]
            th = envs[m["k"]].state[:, envs[m["k"]].num_qubits + 3:]
            th[m["rot_pos"]] = torch.tensor(best, dtype=torch.float32)

    def _spsa_envs_batched(self, envs, opt_states=None, method="SPSA"):
        """SPSA / AdamSPSA for K envs with one CUDA stream per env per iteration.

        Per iteration: one or more independent ±delta probe pairs are batched
        per environment, then all environments are overlapped across streams.
        A separate current-value eval pass tracks the best solution.
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
            bdev = env._batched_vqe.device
            meta.append({
                "k": k, "env": env, "state": st,
                "state_gpu": st.to(bdev),          # cached once; avoids repeated CPU→GPU copies
                "rot_pos": rot_pos, "n": n_params,
                "angles": angles.reshape(1, -1).copy(),   # (1, n)
                "bdev": bdev,
                "a": float(opts.get("a", 0.05)),
                "alpha": float(opts.get("alpha", 0.602)),
                "c": float(opts.get("c", 0.1)),
                "gamma": float(opts.get("gamma", 0.101)),
                "A": float(opts.get("lamda", max(10.0, env.global_iters * 0.1))),
                "b1": float(opts.get("beta_1", 0.9)),
                "b2": float(opts.get("beta_2", 0.999)),
                "spsa_batch": int(env._auto_spsa_batch_size(n_params)),
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
                init_t[i] = m["env"]._batched_vqe.eval_batch(m["state_gpu"], a_t)
        if use_streams:
            torch.cuda.synchronize()
        for i, m in enumerate(meta):
            m["best_val"] = float(init_t[i].item())

        max_iters = max(m["iters"] for m in meta)
        for epoch in range(max_iters):
            # Phase 1+3 merged: probe ±delta rows PLUS current-x row in one eval_batch call.
            # Probe rows  0..2B-1  → gradient estimate.
            # Row 2B (last row)    → energy at current x, used for best-solution tracking.
            # This halves the number of GPU-CPU round-trips vs separate Phase1 / Phase3.
            e_merged = [None] * len(meta)
            for i, (m, stream) in enumerate(zip(meta, streams)):
                if epoch >= m["iters"]:
                    continue
                ck = vc._spsa_ck(epoch, c=m["c"], gamma=m["gamma"])
                spsa_batch = m["spsa_batch"]
                delta = np.random.choice(
                    [-1.0, 1.0], size=(spsa_batch, m["angles"].shape[1])
                ).astype(np.float32)
                m["_delta"] = delta
                m["_ck"] = ck
                probe = np.concatenate([
                    m["angles"] + ck * delta,   # rows 0..B-1
                    m["angles"] - ck * delta,   # rows B..2B-1
                    m["angles"],                # row 2B  ← current x for best tracking
                ], axis=0)  # (2B+1, n)
                ctx = torch.cuda.stream(stream) if stream else contextlib.nullcontext()
                with ctx:
                    probe_t = torch.tensor(probe, dtype=torch.float32, device=m["bdev"])
                    e_merged[i] = m["env"]._batched_vqe.eval_batch(m["state_gpu"], probe_t)
            if use_streams:
                torch.cuda.synchronize()

            # Phase 2: gradient + angle update + best tracking (all from one eval result)
            for i, m in enumerate(meta):
                if epoch >= m["iters"] or e_merged[i] is None:
                    continue
                e_all = e_merged[i].cpu().numpy()   # (2B+1,)
                B = m["spsa_batch"]
                # Best tracking: energy of x_E (row 2B, evaluated before this update)
                val = float(e_all[2 * B])
                if val < m["best_val"]:
                    m["best_val"] = val
                    m["best_angles"] = m["angles"].copy()   # save x_E before overwriting
                # Gradient from probe rows → update to x_{E+1}
                ak = vc._spsa_lr(epoch, a=m["a"], A=m["A"], alpha=m["alpha"])
                ck, delta = m["_ck"], m["_delta"]
                e_plus  = e_all[:B].reshape(-1, 1)
                e_minus = e_all[B:2*B].reshape(-1, 1)
                grad_dirs = ((e_plus - e_minus) / (2.0 * ck)) / delta
                grad = grad_dirs.mean(axis=0, keepdims=True)   # (1, n)
                if use_adam:
                    m["m"] = m["b1"] * m["m"] + (1 - m["b1"]) * grad
                    m["v"] = m["b2"] * m["v"] + (1 - m["b2"]) * grad ** 2
                    mhat = m["m"] / (1 - m["b1"] ** (epoch + 1))
                    vhat = m["v"] / (1 - m["b2"] ** (epoch + 1))
                    m["angles"] = m["angles"] - ak * mhat / (np.sqrt(vhat) + 1e-8)
                else:
                    m["angles"] = m["angles"] - ak * grad

        # Final eval: track energy of the last updated angles (x_K)
        final_t = [None] * len(meta)
        for i, (m, stream) in enumerate(zip(meta, streams)):
            ctx = torch.cuda.stream(stream) if stream else contextlib.nullcontext()
            with ctx:
                a_t = torch.tensor(m["angles"], dtype=torch.float32, device=m["bdev"])
                final_t[i] = m["env"]._batched_vqe.eval_batch(m["state_gpu"], a_t)
        if use_streams:
            torch.cuda.synchronize()
        for i, m in enumerate(meta):
            val = float(final_t[i].item())
            if val < m["best_val"]:
                m["best_val"] = val
                m["best_angles"] = m["angles"].copy()

        for m in meta:
            th = envs[m["k"]].state[:, envs[m["k"]].num_qubits + 3:]
            th[m["rot_pos"]] = torch.tensor(m["best_angles"].flatten(), dtype=torch.float32)
