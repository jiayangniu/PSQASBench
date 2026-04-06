"""
CRLQAS runner for PSQASBench.
All code is self-contained within the PSQASBench package — no sys.path tricks.
"""

import time
from pathlib import Path

import numpy as np
import torch
import wandb

from .utils import get_config, set_seed
from .environment import CircuitEnv
from . import VQE as vc
from . import agents as bench_agents
from .base_runner import BaseRunner


class _Saver:
    """Minimal stats recorder matching the interface CRLQAS agents expect."""

    def __init__(self, result_dir: Path, seed: int):
        self.stats_file = {"train": {}, "test": {}}
        self.exp_seed   = seed
        self.rpath      = result_dir

    def get_new_episode(self, mode, episode_no):
        self.stats_file[mode][episode_no] = {
            "loss": [], "actions": [], "errors": [],
            "errors_noiseless": [], "done_threshold": 0,
            "bond_distance": 0, "nfev": [], "opt_ang": [], "time": [],
        }

    def save_file(self):
        np.save(self.rpath / f"summary_{self.exp_seed}.npy", self.stats_file)


class CRLQASRunner(BaseRunner):
    """
    Run CRLQAS (DQN + N-step + curriculum) on one molecule / seed.

    Args:
        config_path : Path to .cfg in PSQASBench/configs/crlqas/
        mol_path    : Absolute path to the .npz Hamiltonian file
        result_dir  : Directory for result files
        seed        : Random seed
        device      : torch.device (default: cpu)
    """

    def __init__(
        self,
        config_path: Path,
        mol_path: Path,
        result_dir: Path,
        seed: int,
        device: torch.device | None = None,
    ):
        self.config_path = Path(config_path)
        self.device      = device or torch.device("cpu")

        self.conf = get_config(str(self.config_path))

        self.conf["problem"]["mol_data_dir"] = str(Path(mol_path).parent)
        self.conf["problem"]["mol_file"]     = Path(mol_path).name

        # agent is stored on the instance so greedy_episode() can access it
        self._agent = None

        super().__init__(self.conf, mol_path, result_dir, seed)

    @staticmethod
    def _as_bool(v, default=True):
        if v is None:
            return default
        if isinstance(v, bool):
            return v
        if isinstance(v, (int, np.integer)):
            return bool(v)
        if isinstance(v, str):
            s = v.strip().lower()
            if s in {"1", "true", "yes", "y", "on"}:
                return True
            if s in {"0", "false", "no", "n", "off"}:
                return False
        return bool(v)

    # ── helpers ───────────────────────────────────────────────────────────────

    def _modify_state(self, state, env):
        if self.conf["agent"].get("en_state"):
            state = torch.cat(
                (state, torch.tensor([float(env.prev_energy)], dtype=torch.float,
                                     device=self.device))
            )
        return state

    def _save_checkpoint(self, agent, tag: str):
        torch.save(agent.policy_net.state_dict(), self.result_dir / f"{tag}_model.pth")
        torch.save(agent.optim.state_dict(),       self.result_dir / f"{tag}_optim.pth")

    # ── BaseRunner: greedy_episode ─────────────────────────────────────────────

    def greedy_episode(self, env) -> dict:
        """
        Single deterministic (ε=0) rollout with the current DQN policy.

        Called by BaseRunner.periodic_eval() via greedy_rollout_k().
        env.step() is called with train_flag=False to prevent curriculum
        side-effects; the caller (greedy_rollout_k) handles curriculum
        save/restore around all K rollouts.
        """
        agent = self._agent
        accept_err = self.conf["env"]["accept_err"]

        saved_eps = agent.epsilon
        agent.epsilon = 0.0
        agent.policy_net.eval()

        with torch.no_grad():
            state = env.reset()
            state = self._modify_state(state, env)
            itr = 0
            for itr in range(env.num_layers + 1):
                ill   = env.illegal_action_new()
                act, _ = agent.act(state, ill)
                next_state, _, done, _ = env.step(
                    agent.translate[act], train_flag=False
                )
                next_state = self._modify_state(next_state, env)
                state = next_state.clone()
                if done:
                    break

        agent.epsilon = saved_eps
        agent.policy_net.train()

        return {
            "energy_error": float(env.error),
            "cnot_count":   int(env.current_number_of_cnots),
            "success":      int(env.error < accept_err),
            "steps":        itr + 1,
            "op_history":   list(env.op_history),
            "n_qubits":     env.num_qubits,
            "final_state":  env.state.clone(),
        }

    def stochastic_episode(self, env, seed: int) -> dict:
        """
        Single stochastic rollout with current training ε and a fixed seed.
        Used for PCD computation.  Different seeds produce different circuits
        via ε-greedy exploration, making D_struct / D_func meaningful.
        """
        agent = self._agent
        accept_err = self.conf["env"]["accept_err"]

        # seed both numpy and torch for reproducible but varied rollouts
        np.random.seed(seed)
        torch.manual_seed(seed)

        agent.policy_net.eval()   # disable dropout; randomness comes from ε only

        with torch.no_grad():
            state = env.reset()
            state = self._modify_state(state, env)
            itr = 0
            for itr in range(env.num_layers + 1):
                ill   = env.illegal_action_new()
                act, _ = agent.act(state, ill)   # current ε — may explore
                next_state, _, done, _ = env.step(
                    agent.translate[act], train_flag=False
                )
                next_state = self._modify_state(next_state, env)
                state = next_state.clone()
                if done:
                    break

        agent.policy_net.train()

        return {
            "energy_error": float(env.error),
            "cnot_count":   int(env.current_number_of_cnots),
            "success":      int(env.error < accept_err),
            "steps":        itr + 1,
            "op_history":   list(env.op_history),
            "n_qubits":     env.num_qubits,
            "final_state":  env.state.clone(),
        }

    # ── training loop ─────────────────────────────────────────────────────────

    def _one_episode(self, ep, env, agent, global_best,
                     energy_history, cnot_history):
        agent.saver.get_new_episode("train", ep)
        state = env.reset()
        state = self._modify_state(state, env)
        agent.policy_net.train()

        ep_reward = 0.0
        for itr in range(env.num_layers + 1):
            ill   = env.illegal_action_new()
            act, _ = agent.act(state, ill)
            agent.saver.stats_file["train"][ep]["actions"].append(act)

            next_state, reward, done, _ = env.step(agent.translate[act])
            next_state = self._modify_state(next_state, env)

            agent.remember(
                state,
                torch.tensor(act, device=self.device),
                reward,
                next_state,
                torch.tensor(done, device=self.device),
            )
            state = next_state.clone()
            ep_reward += float(reward)
            agent.saver.stats_file["train"][ep]["errors"].append(env.error)
            agent.saver.stats_file["train"][ep]["errors_noiseless"].append(env.error_noiseless)

            energy_now = float(env.energy)
            if energy_now < global_best["energy"]:
                global_best.update({
                    "energy":     energy_now,
                    "episode":    ep,
                    "step":       itr,
                    "error":      abs(energy_now - env.min_energy),
                    "cnot_count": int(env.current_number_of_cnots),
                    "op_history": list(env.op_history),
                    "moments":    list(env.moments),
                    "state":      env.state.clone(),
                })
                self._save_checkpoint(
                    agent,
                    tag=f"best_thresh{self.conf['env']['accept_err']}_seed{self.seed}",
                )
                wandb.log({
                    "global_best/energy":       energy_now,
                    "global_best/energy_error": abs(energy_now - env.min_energy) * 1000,
                    "episode": ep,
                })

            if len(agent.memory) > self.conf["agent"]["batch_size"]:
                loss = agent.replay(self.conf["agent"]["batch_size"])
                wandb.log({"train/policy_loss": loss, "episode": ep})

            if done:
                break

        energy_history.append(float(env.energy))
        cnot_history.append(int(env.current_number_of_cnots))
        return ep_reward, itr + 1

    def _train(self, env, agent, episodes):
        mol_name   = Path(self.mol_path).stem
        config_tag = self.config_path.stem
        eval_every = self.conf["general"].get("eval_every", 1000)
        eval_K     = self.conf["general"].get("eval_K", 20)
        log_every  = max(1, int(self.conf["general"].get("log_every", 10)))
        save_every = max(1, int(self.conf["general"].get("save_every", 200)))

        wandb.init(
            project="PSQASBench",
            entity="jiayangniu14-rmit-university",
            name=f"crlqas_{mol_name}_{config_tag}_seed{self.seed}",
            group=f"crlqas_{mol_name}_{config_tag}",
            config={**self.conf, "config_name": self.config_path.name},
        )
        wandb.define_metric("episode")
        wandb.define_metric("*", step_metric="episode")

        global_best    = {"energy": float("inf"), "episode": None, "step": None,
                          "error": None, "cnot_count": 0, "op_history": None,
                          "moments": None, "state": None}
        energy_history = []
        cnot_history   = []

        t_train_start = time.perf_counter()
        for ep in range(episodes):
            t_ep = time.perf_counter()
            ep_reward, ep_depth = self._one_episode(ep, env, agent, global_best,
                                                    energy_history, cnot_history)
            ep_ms = (time.perf_counter() - t_ep) * 1000
            elapsed = time.perf_counter() - t_train_start
            if ep % log_every == 0:
                print(
                    f"[ep {ep:5d}/{episodes}] "
                    f"t={ep_ms:6.0f}ms  "
                    f"energy={energy_history[-1]:.5f}  "
                    f"best={global_best['energy']:.5f}  "
                    f"err={abs(global_best['energy'] - env.min_energy)*1000:.2f}mHa  "
                    f"cnots={global_best['cnot_count']}  "
                    f"elapsed={elapsed:.0f}s",
                    flush=True,
                )
            wandb.log({
                "train/episode_energy":    energy_history[-1],
                "train/reward_accumulate": ep_reward,
                "train/episode_depth":     ep_depth,
                "episode": ep,
            })

            # ── periodic policy evaluation ─────────────────────────────────
            if eval_every > 0 and ep > 0 and ep % eval_every == 0:
                eval_metrics = self.periodic_eval(ep, env, K=eval_K)
                wandb.log({
                    "eval/sr_at_chem":     eval_metrics["sr_at_chem"],
                    "eval/cnot_at_chem":   eval_metrics["cnot_at_chem"],
                    "eval/best_error_mha": eval_metrics["best_error_mha"],
                    "eval/mean_error_mha": eval_metrics["mean_error_mha"],
                    "eval/mean_cnots":     eval_metrics["mean_cnots"],
                    "eval/D_struct":       eval_metrics["D_struct"],
                    "eval/D_func":         eval_metrics["D_func"],
                    "episode": ep,
                })
                print(
                    f"[eval ep={ep}]  SR={eval_metrics['sr_at_chem']:.2f}  "
                    f"best_err={eval_metrics['best_error_mha']:.2f} mHa  "
                    f"CNOT@chem={eval_metrics['cnot_at_chem']}  "
                    f"D_struct={eval_metrics['D_struct']:.4f}  "
                    f"D_func={eval_metrics['D_func']:.4f}"
                )

            if ep % save_every == 0 and ep > 0:
                agent.saver.save_file()
                if global_best["state"] is not None:
                    np.savez_compressed(
                        self.result_dir / f"global_best_state_{self.seed}.npz",
                        energy=global_best["energy"],
                        episode=global_best["episode"],
                        step=global_best["step"],
                        error=global_best["error"],
                        moments=np.array(global_best["moments"], dtype=np.int32),
                        state=global_best["state"].detach().cpu().numpy().astype(np.float32),
                        op_history=np.array(global_best["op_history"], dtype=object),
                    )

        wandb.finish()
        return global_best, energy_history, cnot_history

    # ── parallel training helpers ──────────────────────────────────────────────

    def _make_nstep_bufs(self, agent, K):
        """Create K per-env N_step_ReplayMemory instances that share agent's
        main replay deque.  Each env gets its own n-step sliding window while
        all writes go into the single shared buffer — no cross-episode bleed
        because the done flag is already handled inside _n_step_return().
        """
        from .agents.DeepQNstep import N_step_ReplayMemory
        n_step     = self.conf["agent"]["n_step"]
        gamma_val  = float(agent.gamma.cpu().item())
        shared_buf = agent.memory.memory          # the underlying deque
        bufs = []
        for _ in range(K):
            b         = N_step_ReplayMemory(self.conf["agent"]["memory_size"],
                                            n_step, gamma_val)
            b.memory  = shared_buf                # all K point to same deque
            bufs.append(b)
        return bufs

    def _eval_env_energies_batched(self, envs):
        """Evaluate current energies for multiple envs with grouped GPU batches.

        Group envs by identical circuit structure (state[:N+3]) so one
        BatchedVQE.eval_batch call can evaluate all angle vectors in that group.
        Falls back to env.get_energy() for phys-noise envs.
        Returns: dict[idx] = (energy, energy_noiseless)
        """
        out = {}
        groups = {}

        for idx, env in enumerate(envs):
            if env.phys_noise:
                continue
            st_struct = env.state[:, : env.num_qubits + 3].contiguous().cpu().numpy().astype(np.uint8)
            key = (env.num_qubits, env.num_layers, st_struct.tobytes())
            groups.setdefault(key, []).append(idx)

        # Noiseless groups: one batched GPU call per unique structure.
        for _, idxs in groups.items():
            ref = envs[idxs[0]]
            bdev = ref._batched_vqe.device
            ref_state = ref.state.to(bdev)
            rot_pos = (ref.state[:, ref.num_qubits: ref.num_qubits + 3] == 1).nonzero(as_tuple=True)
            n_params = len(rot_pos[0])

            if n_params > 0:
                angle_rows = []
                for i in idxs:
                    th = envs[i].state[:, envs[i].num_qubits + 3:]
                    angle_rows.append(np.asarray(th[rot_pos].cpu().detach(), dtype=np.float32))
                angle_batch = torch.tensor(np.stack(angle_rows, axis=0), device=bdev)
            else:
                angle_batch = torch.empty((len(idxs), 0), dtype=torch.float32, device=bdev)

            e_noiseless = ref._batched_vqe.eval_batch(ref_state, angle_batch).detach().cpu().numpy()
            for row, i in enumerate(idxs):
                shot_noise = vc._get_shot_noise(envs[i].weights, envs[i].n_shots)
                out[i] = (float(e_noiseless[row] + shot_noise), float(e_noiseless[row]))

        # Phys-noise path stays with existing scalar evaluator.
        for idx, env in enumerate(envs):
            if idx not in out:
                out[idx] = env.get_energy()

        return out

    def _rotosolve_envs_batched(self, envs, opt_states=None):
        """Run Rotosolve for many envs by batching probes across envs.

        Env states are grouped by identical circuit structure so each group can
        be evaluated by one BatchedVQE kernel per sweep.
        """
        if opt_states is None:
            opt_states = [env.state.clone() for env in envs]

        groups = {}
        for idx, env in enumerate(envs):
            st = opt_states[idx]
            st_struct = st[:, : env.num_qubits + 3].contiguous().cpu().numpy().astype(np.uint8)
            key = (env.num_qubits, env.num_layers, st_struct.tobytes())
            groups.setdefault(key, []).append(idx)

        probe_vals = np.array([0.0, np.pi / 2.0, np.pi], dtype=np.float32)

        for _, idxs in groups.items():
            ref = envs[idxs[0]]
            ref_opt_state = opt_states[idxs[0]]
            rot_pos = (ref_opt_state[:, ref.num_qubits: ref.num_qubits + 3] == 1).nonzero(as_tuple=True)
            n_params = len(rot_pos[0])
            if n_params == 0:
                continue

            bdev = ref._batched_vqe.device
            ref_state = ref_opt_state.to(bdev)
            n_probes_env = 3 * n_params
            G = len(idxs)

            angles = np.zeros((G, n_params), dtype=np.float32)
            for g, env_idx in enumerate(idxs):
                st = opt_states[env_idx]
                th = st[:, envs[env_idx].num_qubits + 3:]
                angles[g] = np.asarray(th[rot_pos].cpu().detach(), dtype=np.float32)

            sweeps = int(getattr(ref, "rotosolve_sweeps", 1))
            for _ in range(sweeps):
                probe_blocks = np.repeat(angles, repeats=n_probes_env, axis=0)
                for g in range(G):
                    base = g * n_probes_env
                    for p in range(n_params):
                        probe_blocks[base + 3 * p : base + 3 * p + 3, p] = probe_vals

                angle_t = torch.tensor(probe_blocks, dtype=torch.float32, device=bdev)
                energies = ref._batched_vqe.eval_batch(ref_state, angle_t).detach().cpu().numpy()

                for g in range(G):
                    e_env = energies[g * n_probes_env : (g + 1) * n_probes_env]
                    for p in range(n_params):
                        e0, epi2, epi = e_env[3 * p], e_env[3 * p + 1], e_env[3 * p + 2]
                        A = (e0 - epi) / 2.0
                        C = (e0 + epi) / 2.0
                        B = epi2 - C
                        angles[g, p] = float(np.arctan2(-B, -A))

            for g, env_idx in enumerate(idxs):
                th = envs[env_idx].state[:, envs[env_idx].num_qubits + 3:]
                th[rot_pos] = torch.tensor(angles[g], dtype=torch.float32)

    def _spsa_envs_batched(self, envs, opt_states=None, method="SPSA"):
        """Run grouped batched SPSA/AdamSPSA across many envs with same structure."""
        if opt_states is None:
            opt_states = [env.state.clone() for env in envs]

        groups = {}
        for idx, env in enumerate(envs):
            st = opt_states[idx]
            st_struct = st[:, : env.num_qubits + 3].contiguous().cpu().numpy().astype(np.uint8)
            key = (env.num_qubits, env.num_layers, st_struct.tobytes())
            groups.setdefault(key, []).append(idx)

        method_name = str(method).upper()
        use_adam = method_name == "ADAMSPSA"

        for _, idxs in groups.items():
            ref = envs[idxs[0]]
            ref_opt_state = opt_states[idxs[0]]
            rot_pos = (ref_opt_state[:, ref.num_qubits: ref.num_qubits + 3] == 1).nonzero(as_tuple=True)
            n_params = len(rot_pos[0])
            if n_params == 0:
                continue

            bdev = ref._batched_vqe.device
            ref_state = ref_opt_state.to(bdev)
            G = len(idxs)

            angles = np.zeros((G, n_params), dtype=np.float32)
            for g, env_idx in enumerate(idxs):
                st = opt_states[env_idx]
                th = st[:, envs[env_idx].num_qubits + 3:]
                angles[g] = np.asarray(th[rot_pos].cpu().detach(), dtype=np.float32)

            opts = getattr(ref, "options", {})
            a      = float(opts.get("a", 0.05))
            alpha  = float(opts.get("alpha", 0.602))
            c      = float(opts.get("c", 0.1))
            gamma  = float(opts.get("gamma", 0.101))
            A      = float(opts.get("lamda", max(10.0, ref.global_iters * 0.1)))
            beta_1 = float(opts.get("beta_1", 0.9))
            beta_2 = float(opts.get("beta_2", 0.999))
            iters  = max(1, int(ref.global_iters))

            angle_t = torch.tensor(angles, dtype=torch.float32, device=bdev)
            best_vals = ref._batched_vqe.eval_batch(ref_state, angle_t).detach().cpu().numpy()
            best_angles = angles.copy()
            m = np.zeros_like(angles, dtype=np.float32)
            v = np.zeros_like(angles, dtype=np.float32)

            for epoch in range(iters):
                ak = vc._spsa_lr(epoch, a=a, A=A, alpha=alpha)
                ck = vc._spsa_ck(epoch, c=c, gamma=gamma)
                delta = np.random.choice([-1.0, 1.0], size=angles.shape).astype(np.float32)

                plus_t = torch.tensor(angles + ck * delta, dtype=torch.float32, device=bdev)
                minus_t = torch.tensor(angles - ck * delta, dtype=torch.float32, device=bdev)
                probe_t = torch.cat((plus_t, minus_t), dim=0)
                probe_e = ref._batched_vqe.eval_batch(ref_state, probe_t).detach().cpu().numpy()
                e_plus, e_minus = probe_e[:G], probe_e[G:]

                grad_scale = ((e_plus - e_minus) / (2.0 * ck)).astype(np.float32)[:, None]
                grad = grad_scale / delta

                if use_adam:
                    m = beta_1 * m + (1.0 - beta_1) * grad
                    v = beta_2 * v + (1.0 - beta_2) * (grad ** 2)
                    mhat = m / (1.0 - beta_1 ** (epoch + 1))
                    vhat = v / (1.0 - beta_2 ** (epoch + 1))
                    step = mhat / (np.sqrt(vhat) + 1e-8)
                    angles = angles - ak * step
                else:
                    angles = angles - ak * grad

                curr_t = torch.tensor(angles, dtype=torch.float32, device=bdev)
                curr_vals = ref._batched_vqe.eval_batch(ref_state, curr_t).detach().cpu().numpy()
                improved = curr_vals < best_vals
                best_vals[improved] = curr_vals[improved]
                best_angles[improved] = angles[improved]

            for g, env_idx in enumerate(idxs):
                th = envs[env_idx].state[:, envs[env_idx].num_qubits + 3:]
                th[rot_pos] = torch.tensor(best_angles[g], dtype=torch.float32)

    def _train_parallel(self, envs, agent, total_episodes):
        """Parallel training loop: batch policy inference + grouped GPU energy eval.

        K envs advance one step each round. Actions are produced in one
        policy forward pass; energy evaluation is deferred and then grouped by
        circuit structure so each group is evaluated via one BatchedVQE call.

        Invariant: batch_size must be divisible by K so that every K new
        transitions added to the replay buffer correspond to exactly
        batch_size // K replay gradient steps when the buffer is full.
        """
        K          = len(envs)
        batch_size = self.conf["agent"]["batch_size"]
        eval_every = self.conf["general"].get("eval_every", 1000)
        eval_K     = self.conf["general"].get("eval_K", 20)
        log_every  = max(1, int(self.conf["general"].get("log_every", 10)))
        save_every = max(1, int(self.conf["general"].get("save_every", 200)))

        nstep_bufs = self._make_nstep_bufs(agent, K)

        # initialise all envs
        states = [self._modify_state(env.reset(), env) for env in envs]

        # After reset all K states are identical (zero circuit).  When ε is
        # low, act_batch() would return the same greedy action for every env,
        # filling the replay buffer with duplicate transitions.  Force random
        # exploration on the FIRST step of every newly-reset env to guarantee
        # structural diversity from step 1.
        just_reset = [True] * K

        global_best = {"energy": float("inf"), "episode": None, "step": None,
                       "error": None, "cnot_count": 0, "op_history": None,
                       "moments": None, "state": None}
        energy_history, cnot_history = [], []
        total_ep = 0

        mol_name = Path(self.mol_path).stem
        config_tag = self.config_path.stem
        wandb.init(
            project="PSQASBench",
            entity="jiayangniu14-rmit-university",
            name=f"crlqas_par{K}_{mol_name}_{config_tag}_seed{self.seed}",
            group=f"crlqas_{mol_name}_{config_tag}",
            config={**self.conf, "num_parallel_envs": K, "config_name": self.config_path.name},
        )
        wandb.define_metric("episode")
        wandb.define_metric("*", step_metric="episode")

        # Dedicated eval env — periodic_eval runs K greedy episodes that modify
        # the env's moments/step_counter.  Using a training env would corrupt its
        # circuit state between rounds (greedy_rollout_k only saves curriculum).
        _eval_env = CircuitEnv(self.conf, device=self.device, use_gpu_state=False)

        t_train_start = time.perf_counter()
        cfg_non_local = self.conf.get("non_local_opt", {})
        global_rotosolve_enabled = self._as_bool(
            cfg_non_local.get("global_batched_rotosolve", 1), default=True
        )
        use_global_batched_rotosolve = global_rotosolve_enabled and all(
            (env.optim_method == "scipy_each_step" and env.optim_alg == "Rotosolve" and not env.phys_noise)
            for env in envs
        )
        global_spsa_enabled = self._as_bool(
            cfg_non_local.get("global_batched_spsa", 1), default=True
        )
        use_global_batched_spsa = global_spsa_enabled and all(
            (env.optim_method == "scipy_each_step" and str(env.optim_alg).upper() in {"SPSA", "ADAMSPSA"} and not env.phys_noise)
            for env in envs
        )
        use_global_batched_optimizer = (
            use_global_batched_rotosolve or use_global_batched_spsa
        )
        optimizer_mode = None
        if use_global_batched_rotosolve:
            optimizer_mode = "Rotosolve"
        elif use_global_batched_spsa:
            optimizer_mode = str(envs[0].optim_alg)

        if optimizer_mode is not None:
            print(
                f"[parallel-opt] mode={optimizer_mode}  envs={K}  "
                f"global_iters={int(envs[0].global_iters)}",
                flush=True,
            )

        round_idx = 0
        last_progress_log = t_train_start

        while total_ep < total_episodes:
            round_idx += 1
            # ── 1. batch act: one GPU forward pass for all K states ──────────
            agent.policy_net.train()
            ill_list    = [env.illegal_action_new() for env in envs]
            act_results = agent.act_batch(states, ill_list)
            # Force random first step after reset: same initial state → same
            # greedy action → duplicate experience.  One random gate breaks it.
            for k in range(K):
                if just_reset[k]:
                    a = torch.randint(agent.action_size, (1,)).item()
                    while a in ill_list[k]:
                        a = torch.randint(agent.action_size, (1,)).item()
                    act_results[k] = (a, True)
                    just_reset[k] = False

            # ── 2. Apply actions, run (possibly global) optimiser, then energies ──
            step_states = {}
            opt_ref_states = {}
            for k in range(K):
                step_states[k] = envs[k].step_deferred_energy(
                    agent.translate[act_results[k][0]],
                    run_optimizer=not use_global_batched_optimizer,
                )
                # Capture post-action structure for grouped batched optimisers.
                opt_ref_states[k] = envs[k].state.clone()

            if use_global_batched_rotosolve:
                self._rotosolve_envs_batched(
                    envs, opt_states=[opt_ref_states[k] for k in range(K)]
                )
                for k in range(K):
                    step_states[k] = envs[k].state.clone()
            elif use_global_batched_spsa:
                self._spsa_envs_batched(
                    envs,
                    opt_states=[opt_ref_states[k] for k in range(K)],
                    method=envs[0].optim_alg,
                )
                for k in range(K):
                    step_states[k] = envs[k].state.clone()

            energies = self._eval_env_energies_batched(envs)
            outcomes = {}
            for k in range(K):
                e, e0 = energies[k]
                outcomes[k] = envs[k].finalize_step_with_energy(
                    action=agent.translate[act_results[k][0]],
                    next_state=step_states[k],
                    energy=e,
                    energy_noiseless=e0,
                )

            # ── 3. collect experiences, handle episode boundaries ─────────────
            new_states = list(states)
            for k in range(K):
                ns, reward, done, _ = outcomes[k]
                ns_mod = self._modify_state(ns, envs[k])

                nstep_bufs[k].push(
                    states[k],
                    torch.tensor(act_results[k][0], device=self.device),
                    reward,
                    ns_mod,
                    torch.tensor(done, device=self.device),
                )

                energy_now = float(envs[k].energy)
                if energy_now < global_best["energy"]:
                    global_best.update({
                        "energy":     energy_now,
                        "episode":    total_ep,
                        "step":       envs[k].step_counter,
                        "error":      abs(energy_now - envs[k].min_energy),
                        "cnot_count": int(envs[k].current_number_of_cnots),
                        "op_history": list(envs[k].op_history),
                        "moments":    list(envs[k].moments),
                        "state":      envs[k].state.clone(),
                    })
                    self._save_checkpoint(
                        agent,
                        tag=f"best_thresh{self.conf['env']['accept_err']}_seed{self.seed}",
                    )
                    wandb.log({
                        "global_best/energy":       energy_now,
                        "global_best/energy_error": abs(energy_now - envs[k].min_energy) * 1000,
                        "episode": total_ep,
                    })

                if done:
                    energy_history.append(float(envs[k].energy))
                    cnot_history.append(int(envs[k].current_number_of_cnots))
                    elapsed = time.perf_counter() - t_train_start
                    if total_ep % log_every == 0:
                        print(
                            f"[ep {total_ep:5d}/{total_episodes}  env{k}] "
                            f"energy={energy_history[-1]:.5f}  "
                            f"best={global_best['energy']:.5f}  "
                            f"err={abs(global_best['energy'] - envs[k].min_energy)*1000:.2f}mHa  "
                            f"cnots={global_best['cnot_count']}  "
                            f"elapsed={elapsed:.0f}s",
                            flush=True,
                        )
                    wandb.log({
                        "train/episode_energy": energy_history[-1],
                        "train/episode_depth":  envs[k].step_counter + 1,
                        "episode": total_ep,
                    })
                    total_ep += 1
                    if total_ep % save_every == 0:
                        agent.saver.save_file()
                        if global_best["state"] is not None:
                            np.savez_compressed(
                                self.result_dir / f"global_best_state_{self.seed}.npz",
                                energy=global_best["energy"],
                                episode=global_best["episode"],
                                step=global_best["step"],
                                error=global_best["error"],
                                moments=np.array(global_best["moments"], dtype=np.int32),
                                state=global_best["state"].detach().cpu().numpy().astype(np.float32),
                                op_history=np.array(global_best["op_history"], dtype=object),
                            )

                    if eval_every > 0 and total_ep > 0 and total_ep % eval_every == 0:
                        eval_metrics = self.periodic_eval(total_ep, _eval_env, K=eval_K)
                        wandb.log({
                            "eval/sr_at_chem":     eval_metrics["sr_at_chem"],
                            "eval/cnot_at_chem":   eval_metrics["cnot_at_chem"],
                            "eval/best_error_mha": eval_metrics["best_error_mha"],
                            "eval/mean_error_mha": eval_metrics["mean_error_mha"],
                            "eval/mean_cnots":     eval_metrics["mean_cnots"],
                            "eval/D_struct":       eval_metrics["D_struct"],
                            "eval/D_func":         eval_metrics["D_func"],
                            "episode": total_ep,
                        })

                    new_states[k] = self._modify_state(envs[k].reset(), envs[k])
                    just_reset[k] = True
                else:
                    new_states[k] = ns_mod

            states = new_states

            elapsed = time.perf_counter() - t_train_start
            if (
                optimizer_mode is not None
                and elapsed - (last_progress_log - t_train_start) >= 60.0
            ):
                max_depth = max(int(env.step_counter) for env in envs)
                min_depth = min(int(env.step_counter) for env in envs)
                print(
                    f"[parallel-opt] mode={optimizer_mode}  rounds={round_idx}  "
                    f"episodes_done={total_ep}/{total_episodes}  "
                    f"env_depth_range={min_depth}-{max_depth}  elapsed={elapsed:.0f}s",
                    flush=True,
                )
                last_progress_log = time.perf_counter()

            # ── 4. DQN replay: one update per round (K new transitions added) ─
            if len(agent.memory) > batch_size:
                loss = agent.replay(batch_size)
                wandb.log({"train/policy_loss": loss, "episode": total_ep})

        wandb.finish()
        return global_best, energy_history, cnot_history

    # ── public interface ──────────────────────────────────────────────────────

    def run(self) -> dict:
        set_seed(self.seed)
        torch.set_num_threads(1)

        env   = CircuitEnv(self.conf, device=self.device)
        agent = (bench_agents
                 .__dict__[self.conf["agent"]["agent_type"]]
                 .__dict__[self.conf["agent"]["agent_class"]]
                 (self.conf, env.action_size, env.state_size, self.device))
        agent.saver  = _Saver(self.result_dir, self.seed)
        self._agent  = agent   # expose for greedy_episode()

        global_best, energy_history, cnot_history = self.train_with_parallel_entry(
            env=env,
            agent=agent,
            episodes=self.conf["general"]["episodes"],
            train_single_fn=self._train,
            make_env_fn=lambda: CircuitEnv(self.conf, device=self.device, use_gpu_state=False),
            train_parallel_fn=self._train_parallel,
            require_batch_divisible=True,
        )
        agent.saver.save_file()

        accept_err = self.conf["env"]["accept_err"]
        best_err   = abs(global_best["error"]) if global_best["error"] is not None else float("inf")

        result = {
            "method":         "CRLQAS",
            "seed":           self.seed,
            "best_energy":    global_best["energy"],
            "energy_error":   best_err,
            "cnot_count":     global_best["cnot_count"],
            "circuit_depth":  global_best["step"] if global_best["step"] is not None else -1,
            "nfev":           -1,
            "success":        int(best_err < accept_err),
            "energy_history": energy_history,
            "cnot_history":   cnot_history,
            "exact_energy":   self.exact_energy,
            "accept_err":     accept_err,
        }

        out = self.save_result(result)
        print(f"[CRLQASRunner] seed={self.seed}  error={best_err*1000:.3f} mHa  "
              f"cnots={global_best['cnot_count']}  success={bool(result['success'])}  "
              f"→ {out}")
        return result
