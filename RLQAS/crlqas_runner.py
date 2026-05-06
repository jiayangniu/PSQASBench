"""
CRLQAS runner for PSQASBench.
All code is self-contained within the PSQASBench package — no sys.path tricks.
"""

import os
import time
from pathlib import Path

import numpy as np
import torch
import wandb

from .result_logger import ResultLogger
from .utils import count_rotation_gates, get_config, set_seed, set_torch_seed
from .environment import CircuitEnv
from . import VQE as vc
from . import agents as bench_agents
from .base_runner import BaseRunner


class _Saver:
    """Minimal stats recorder matching the interface CRLQAS agents expect."""

    def __init__(self, result_dir: Path, seed: int, enabled: bool = True):
        self.stats_file = {"train": {}, "test": {}}
        self.exp_seed   = seed
        self.rpath      = result_dir
        self.enabled    = enabled

    def get_new_episode(self, mode, episode_no):
        if not self.enabled:
            return
        self.stats_file[mode][episode_no] = {
            "loss": [], "actions": [], "errors": [],
            "errors_noiseless": [], "done_threshold": 0,
            "bond_distance": 0, "nfev": [], "opt_ang": [], "time": [],
        }

    def save_file(self):
        if not self.enabled:
            return
        if not self.stats_file["train"] and not self.stats_file["test"]:
            return
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
        mol_name: str | None = None,
        save_summary_detailed: int | None = None,
        use_wandb: int | None = None,
    ):
        self.config_path = Path(config_path)
        self.device      = device or torch.device("cpu")
        self.mol_name    = mol_name

        self.conf = get_config(str(self.config_path))
        if save_summary_detailed is not None:
            self.conf.setdefault("general", {})["save_summary_detailed"] = int(save_summary_detailed)
        if use_wandb is not None:
            self.conf.setdefault("general", {})["use_wandb"] = int(use_wandb)
        self.save_summary_detailed = bool(int(self.conf.get("general", {}).get("save_summary_detailed", 0)))
        self.use_wandb = bool(int(self.conf.get("general", {}).get("use_wandb", 0)))
        self.runtime_overrides = {}
        if save_summary_detailed is not None:
            self.runtime_overrides["save_summary_detailed"] = int(save_summary_detailed)
        if use_wandb is not None:
            self.runtime_overrides["use_wandb"] = int(use_wandb)

        self.conf["problem"]["mol_data_dir"] = str(Path(mol_path).parent)
        self.conf["problem"]["mol_file"]     = Path(mol_path).name

        # agent is stored on the instance so greedy_episode() can access it
        self._agent = None
        self.artifacts = None

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

    def _wandb_init(self, run_name: str, group: str, config_dict: dict):
        general = self.conf.get("general", {})
        project = os.environ.get("WANDB_PROJECT") or general.get("wandb_project") or "PSQASBench"
        entity = os.environ.get("WANDB_ENTITY") or general.get("wandb_entity") or ""

        kwargs = dict(
            project=project,
            name=run_name,
            group=group,
            mode=("online" if self.use_wandb else "disabled"),
            config=config_dict,
        )
        if str(entity).strip():
            kwargs["entity"] = str(entity).strip()
        wandb.init(**kwargs)

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

    def _mol_key(self) -> str:
        if self.mol_name:
            return self.mol_name
        return self.result_dir.parents[1].name

    def _init_artifacts(self):
        self.artifacts = ResultLogger(
            self.result_dir,
            self.config_path,
            method="crlqas",
            mol=self._mol_key(),
            seed=self.seed,
            device=str(self.device),
            exact_energy=self.exact_energy,
            accept_err=float(self.conf["env"]["accept_err"]),
            analysis_save_threshold=float(self._analysis_save_threshold(self.conf)),
            save_summary_detailed=self.save_summary_detailed,
            runtime_overrides=self.runtime_overrides,
        )

    def _bind_wandb_run(self):
        if self.artifacts is not None and wandb.run is not None:
            self.artifacts.set_wandb_run(wandb.run.name, wandb.run.id)

    def _record_global_best(self, global_best, env, episode: int, step: int, agent):
        energy_now = float(env.energy)
        rotation_count = int(count_rotation_gates(env.op_history))
        depth = int(step) + 1
        global_best.update({
            "energy": energy_now,
            "episode": episode,
            "step": step,
            "error": abs(energy_now - env.min_energy),
            "cnot_count": int(env.current_number_of_cnots),
            "rotation_count": rotation_count,
            "depth": depth,
            "op_history": list(env.op_history),
            "moments": list(env.moments),
            "state": env.state.clone(),
        })
        self._save_checkpoint(
            agent,
            tag=f"best_thresh{self.conf['env']['accept_err']}_seed{self.seed}",
        )
        wandb.log({
            "global_best/energy": energy_now,
            "global_best/energy_error": abs(energy_now - env.min_energy) * 1000.0,
            "global_best/depth": depth,
            "global_best/cnot_count": int(env.current_number_of_cnots),
            "global_best/rotation_count": rotation_count,
            "episode": episode,
        })
        if self.artifacts is not None:
            self.artifacts.update_best_train({
                "episode": episode,
                "step": step,
                "energy": energy_now,
                "energy_error": abs(energy_now - env.min_energy),
                "depth": depth,
                "cnot_count": int(env.current_number_of_cnots),
                "rotation_count": rotation_count,
                "op_history": list(env.op_history),
            })

    def _append_policy_loss(self, episode: int, step: int, loss: float):
        if self.artifacts is not None:
            self.artifacts.append_policy_loss(
                episode=episode,
                step=step,
                policy_loss=float(loss),
            )

    def _episode_summary_record(self, *, episode: int, env, episode_return: float,
                                episode_time_sec: float, is_global_best: bool, epsilon_end):
        depth = int(env.step_counter) + 1
        rotation_count = int(count_rotation_gates(env.op_history))
        return {
            "episode": episode,
            "final_energy_ha": float(env.energy),
            "final_energy_error_ha": float(env.error),
            "episode_return": float(episode_return),
            "depth": depth,
            "cnot_count": int(env.current_number_of_cnots),
            "rotation_count": rotation_count,
            "n_steps": depth,
            "epsilon_end": epsilon_end,
            "episode_time_sec": float(episode_time_sec),
            "is_global_best": int(bool(is_global_best)),
        }

    # ── BaseRunner: greedy_episode ─────────────────────────────────────────────

    def greedy_episode(self, env) -> dict:
        """
        Single deterministic (ε=0) rollout with the current DQN policy.

        env.step() is called with train_flag=False to prevent curriculum
        side-effects; the caller handles curriculum save/restore around all K rollouts.
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
            "energy":       float(env.energy),
            "energy_error": float(env.error),
            "cnot_count":   int(env.current_number_of_cnots),
            "rotation_count": int(count_rotation_gates(env.op_history)),
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
        set_torch_seed(seed, device=self.device)

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
            "energy":       float(env.energy),
            "energy_error": float(env.error),
            "cnot_count":   int(env.current_number_of_cnots),
            "rotation_count": int(count_rotation_gates(env.op_history)),
            "success":      int(env.error < accept_err),
            "steps":        itr + 1,
            "op_history":   list(env.op_history),
            "n_qubits":     env.num_qubits,
            "final_state":  env.state.clone(),
        }

    # ── training loop ─────────────────────────────────────────────────────────

    def _one_episode(self, ep, env, agent, global_best,
                     energy_history, cnot_history):
        if agent.saver.enabled:
            agent.saver.get_new_episode("train", ep)
        state = env.reset()
        state = self._modify_state(state, env)
        agent.policy_net.train()

        ep_reward = 0.0
        analysis_save_threshold_ha = float(self._analysis_save_threshold(self.conf))
        trace = {
            "actions": [],
            "energies": [],
            "energy_errors": [],
            "rewards": [],
            "analysis_snapshots": [],
        }
        episode_hit_global_best = False
        for itr in range(env.num_layers + 1):
            ill   = env.illegal_action_new()
            act, _ = agent.act(state, ill)
            trace["actions"].append(int(act))
            if agent.saver.enabled:
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
            prev_error_ha = (
                float(trace["energy_errors"][-1])
                if trace["energy_errors"]
                else float("inf")
            )
            trace["energies"].append(float(env.energy))
            trace["energy_errors"].append(float(env.error))
            trace["rewards"].append(float(reward))
            self._maybe_append_analysis_snapshot(
                trace,
                env,
                analysis_save_threshold_ha=analysis_save_threshold_ha,
                prev_error_ha=prev_error_ha,
            )
            if agent.saver.enabled:
                agent.saver.stats_file["train"][ep]["errors"].append(env.error)
                agent.saver.stats_file["train"][ep]["errors_noiseless"].append(env.error_noiseless)

            energy_now = float(env.energy)
            if energy_now < global_best["energy"]:
                self._record_global_best(global_best, env, ep, itr, agent)
                episode_hit_global_best = True

            if len(agent.memory) > self.conf["agent"]["batch_size"]:
                loss = agent.replay(self.conf["agent"]["batch_size"])
                wandb.log({"train/policy_loss": loss, "episode": ep})
                self._append_policy_loss(ep, itr, loss)

            if done:
                break

        energy_history.append(float(env.energy))
        cnot_history.append(int(env.current_number_of_cnots))
        return ep_reward, itr + 1, trace, episode_hit_global_best

    def _train(self, env, agent, episodes):
        mol_name   = Path(self.mol_path).stem
        config_tag = self.config_path.stem
        eval_every = self.conf["general"].get("eval_every", 1000)
        eval_K     = self.conf["general"].get("eval_K", 20)
        log_every  = max(1, int(self.conf["general"].get("log_every", 10)))
        save_every = max(1, int(self.conf["general"].get("save_every", 200)))

        _ham_type  = self.conf["problem"].get("ham_type", "mol")
        _nq        = self.conf["env"]["num_qubits"]
        _mol_short = f"{_ham_type}_{_nq}q"
        _optim     = self.conf.get("non_local_opt", {}).get("optim_alg", "noopt")
        _device    = self.device.type  # "cpu" or "cuda"
        _run_name  = f"crlqas_{_mol_short}_{_optim}_{_device}_seed{self.seed}"
        _group     = f"crlqas_{_mol_short}_{_optim}_{_device}"

        self._wandb_init(
            _run_name,
            _group,
            {**self.conf, "config_name": self.config_path.name},
        )
        self._bind_wandb_run()
        wandb.define_metric("episode")
        wandb.define_metric("*", step_metric="episode")

        global_best    = {"energy": float("inf"), "episode": None, "step": None,
                          "error": None, "cnot_count": 0, "op_history": None,
                          "rotation_count": 0, "depth": 0,
                          "moments": None, "state": None}
        energy_history = []
        cnot_history   = []

        t_train_start = time.perf_counter()
        for ep in range(episodes):
            t_ep = time.perf_counter()
            ep_reward, ep_depth, ep_trace, ep_hit_global_best = self._one_episode(
                ep, env, agent, global_best, energy_history, cnot_history
            )
            ep_sec = time.perf_counter() - t_ep
            ep_ms = ep_sec * 1000
            elapsed = time.perf_counter() - t_train_start
            epsilon_end = float(getattr(agent, "epsilon", float("nan")))
            rotation_count = int(count_rotation_gates(env.op_history))
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
                "train/episode_energy_error": float(env.error) * 1000.0,
                "train/reward_accumulate": ep_reward,
                "train/episode_depth":     ep_depth,
                "train/episode_cnot_count": int(env.current_number_of_cnots),
                "train/episode_rotation_count": rotation_count,
                "train/episode_time_sec": ep_sec,
                "train/epsilon": epsilon_end,
                "episode": ep,
            })
            if self.artifacts is not None:
                self.artifacts.append_episode_summary(
                    self._episode_summary_record(
                        episode=ep,
                        env=env,
                        episode_return=ep_reward,
                        episode_time_sec=ep_sec,
                        is_global_best=ep_hit_global_best,
                        epsilon_end=epsilon_end,
                    )
                )
                self.artifacts.append_episode_trace(ep, ep_trace)

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
                if self.artifacts is not None:
                    self.artifacts.update_eval(eval_metrics)
                print(
                    f"[eval ep={ep}]  SR={eval_metrics['sr_at_chem']:.2f}  "
                    f"best_err={eval_metrics['best_error_mha']:.2f} mHa  "
                    f"CNOT@chem={eval_metrics['cnot_at_chem']}  "
                    f"D_struct={eval_metrics['D_struct']:.4f}  "
                    f"D_func={eval_metrics['D_func']:.4f}"
                )

            if ep % save_every == 0 and ep > 0:
                if self.save_summary_detailed:
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
        ep_returns = [0.0] * K
        ep_starts = [time.perf_counter()] * K
        ep_hit_global_best = [False] * K
        analysis_save_threshold_ha = float(self._analysis_save_threshold(self.conf))
        ep_traces = [{
            "actions": [],
            "energies": [],
            "energy_errors": [],
            "rewards": [],
            "analysis_snapshots": [],
        } for _ in range(K)]

        global_best = {"energy": float("inf"), "episode": None, "step": None,
                       "error": None, "cnot_count": 0, "op_history": None,
                       "rotation_count": 0, "depth": 0,
                       "moments": None, "state": None}
        energy_history, cnot_history = [], []
        total_ep = 0

        mol_name   = Path(self.mol_path).stem
        config_tag = self.config_path.stem
        _ham_type  = self.conf["problem"].get("ham_type", "mol")
        _nq        = self.conf["env"]["num_qubits"]
        _mol_short = f"{_ham_type}_{_nq}q"
        _optim     = self.conf.get("non_local_opt", {}).get("optim_alg", "noopt")
        _device    = self.device.type
        _run_name  = f"crlqas_{_mol_short}_{_optim}_{_device}_seed{self.seed}"
        _group     = f"crlqas_{_mol_short}_{_optim}_{_device}"
        self._wandb_init(
            _run_name,
            _group,
            {**self.conf, "num_parallel_envs": K, "config_name": self.config_path.name},
        )
        self._bind_wandb_run()
        wandb.define_metric("episode")
        wandb.define_metric("*", step_metric="episode")

        # Dedicated eval env — periodic_eval runs K greedy episodes that modify
        # the env's moments/step_counter.  Using a training env would corrupt its
        # circuit state between rounds (greedy_rollout_k only saves curriculum).
        _eval_env = CircuitEnv(self.conf, device=self.device, use_gpu_state=False,
                               shared_bvqe=envs[0]._batched_vqe)

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
        global_psr_enabled = self._as_bool(
            cfg_non_local.get("global_batched_psr", 1), default=True
        )
        use_global_batched_psr = global_psr_enabled and all(
            (env.optim_method == "scipy_each_step" and str(env.optim_alg).upper() == "PSRADAM" and not env.phys_noise)
            for env in envs
        )
        use_global_batched_optimizer = (
            use_global_batched_rotosolve or use_global_batched_spsa or use_global_batched_psr
        )
        optimizer_mode = None
        if use_global_batched_rotosolve:
            optimizer_mode = "Rotosolve"
        elif use_global_batched_spsa:
            optimizer_mode = str(envs[0].optim_alg)
        elif use_global_batched_psr:
            optimizer_mode = "PSRAdam"

        if optimizer_mode is not None:
            if use_global_batched_rotosolve:
                budget_str = f"sweeps={int(getattr(envs[0], 'rotosolve_sweeps', 1))}"
            elif str(optimizer_mode).upper() in {"SPSA", "ADAMSPSA"}:
                budget_str = f"global_iters={int(envs[0].global_iters)}  spsa_batch=auto"
            else:
                budget_str = f"global_iters={int(envs[0].global_iters)}"
            print(
                f"[parallel-opt] mode={optimizer_mode}  envs={K}  "
                f"{budget_str}",
                flush=True,
            )

        round_idx = 0

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
                    # Preserve any configured exploration bias (for example
                    # qubit-importance weighting) on the forced-random first step.
                    if hasattr(agent, "_explore"):
                        a = agent._explore(ill_list[k])
                    else:
                        a = torch.randint(agent.action_size, (1,)).item()
                        while a in ill_list[k]:
                            a = torch.randint(agent.action_size, (1,)).item()
                    act_results[k] = (a, True)
                    just_reset[k] = False
                ep_traces[k]["actions"].append(int(act_results[k][0]))

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
            elif use_global_batched_psr:
                self._psr_adam_envs_batched(
                    envs, opt_states=[opt_ref_states[k] for k in range(K)]
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
                ep_returns[k] += float(reward)
                ep_traces[k]["rewards"].append(float(reward))
                prev_error_ha = (
                    float(ep_traces[k]["energy_errors"][-1])
                    if ep_traces[k]["energy_errors"]
                    else float("inf")
                )
                ep_traces[k]["energies"].append(float(envs[k].energy))
                ep_traces[k]["energy_errors"].append(float(envs[k].error))
                self._maybe_append_analysis_snapshot(
                    ep_traces[k],
                    envs[k],
                    analysis_save_threshold_ha=analysis_save_threshold_ha,
                    prev_error_ha=prev_error_ha,
                )

                nstep_bufs[k].push(
                    states[k],
                    torch.tensor(act_results[k][0], device=self.device),
                    reward,
                    ns_mod,
                    torch.tensor(done, device=self.device),
                )

                energy_now = float(envs[k].energy)
                if energy_now < global_best["energy"]:
                    self._record_global_best(global_best, envs[k], total_ep, envs[k].step_counter, agent)
                    ep_hit_global_best[k] = True

                if done:
                    episode_idx = total_ep
                    ep_sec = time.perf_counter() - ep_starts[k]
                    epsilon_end = float(getattr(agent, "epsilon", float("nan")))
                    rotation_count = int(count_rotation_gates(envs[k].op_history))
                    energy_history.append(float(envs[k].energy))
                    cnot_history.append(int(envs[k].current_number_of_cnots))
                    elapsed = time.perf_counter() - t_train_start
                    if episode_idx % log_every == 0:
                        print(
                            f"[ep {episode_idx:5d}/{total_episodes}  env{k}] "
                            f"energy={energy_history[-1]:.5f}  "
                            f"best={global_best['energy']:.5f}  "
                            f"err={abs(global_best['energy'] - envs[k].min_energy)*1000:.2f}mHa  "
                            f"cnots={global_best['cnot_count']}  "
                            f"elapsed={elapsed:.0f}s",
                            flush=True,
                        )
                    wandb.log({
                        "train/episode_energy": energy_history[-1],
                        "train/episode_energy_error": float(envs[k].error) * 1000.0,
                        "train/reward_accumulate": ep_returns[k],
                        "train/episode_depth": envs[k].step_counter + 1,
                        "train/episode_cnot_count": int(envs[k].current_number_of_cnots),
                        "train/episode_rotation_count": rotation_count,
                        "train/episode_time_sec": ep_sec,
                        "train/epsilon": epsilon_end,
                        "episode": episode_idx,
                    })
                    if self.artifacts is not None:
                        self.artifacts.append_episode_summary(
                            self._episode_summary_record(
                                episode=episode_idx,
                                env=envs[k],
                                episode_return=ep_returns[k],
                                episode_time_sec=ep_sec,
                                is_global_best=ep_hit_global_best[k],
                                epsilon_end=epsilon_end,
                            )
                        )
                        self.artifacts.append_episode_trace(episode_idx, ep_traces[k])
                    total_ep += 1
                    if total_ep % save_every == 0:
                        if self.save_summary_detailed:
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
                        # Sync curriculum state from a training env so greedy episodes
                        # use the current (evolved) done_threshold, not the initial one.
                        import copy as _copy
                        _eval_env.done_threshold  = envs[0].done_threshold
                        _eval_env.curriculum      = _copy.deepcopy(envs[0].curriculum)
                        _eval_env.curriculum_dict = _copy.deepcopy(envs[0].curriculum_dict)
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
                        if self.artifacts is not None:
                            self.artifacts.update_eval(eval_metrics)

                    new_states[k] = self._modify_state(envs[k].reset(), envs[k])
                    just_reset[k] = True
                    ep_returns[k] = 0.0
                    ep_starts[k] = time.perf_counter()
                    ep_hit_global_best[k] = False
                    ep_traces[k] = {
                        "actions": [],
                        "energies": [],
                        "energy_errors": [],
                        "rewards": [],
                        "analysis_snapshots": [],
                    }
                else:
                    new_states[k] = ns_mod

            states = new_states

            # ── 4. DQN replay: one update per round (K new transitions added) ─
            if len(agent.memory) > batch_size:
                loss = agent.replay(batch_size)
                wandb.log({"train/policy_loss": loss, "episode": total_ep})
                self._append_policy_loss(total_ep, round_idx, loss)

        wandb.finish()
        return global_best, energy_history, cnot_history

    # ── public interface ──────────────────────────────────────────────────────

    def run(self) -> dict:
        set_seed(self.seed, device=self.device)
        torch.set_num_threads(1)
        run_start = time.perf_counter()
        self._init_artifacts()

        # Keep the Qulacs state on CPU even for the root env so one run does
        # not silently touch the default GPU in addition to the requested
        # BatchedVQE device (cuda:X).
        env   = CircuitEnv(self.conf, device=self.device, use_gpu_state=False)
        agent = (bench_agents
                 .__dict__[self.conf["agent"]["agent_type"]]
                 .__dict__[self.conf["agent"]["agent_class"]]
                 (self.conf, env.action_size, env.state_size, self.device))
        agent.saver  = _Saver(self.result_dir, self.seed, enabled=self.save_summary_detailed)
        self._agent  = agent   # expose for greedy_episode()

        _shared_bvqe = env._batched_vqe
        global_best, energy_history, cnot_history = self.train_with_parallel_entry(
            env=env,
            agent=agent,
            episodes=self.conf["general"]["episodes"],
            train_single_fn=self._train,
            make_env_fn=lambda: CircuitEnv(self.conf, device=self.device, use_gpu_state=False,
                                           shared_bvqe=_shared_bvqe),
            train_parallel_fn=self._train_parallel,
            require_batch_divisible=True,
        )
        if self.save_summary_detailed:
            agent.saver.save_file()

        accept_err = self.conf["env"]["accept_err"]
        best_err   = abs(global_best["error"]) if global_best["error"] is not None else float("inf")

        result = {
            "method":         "CRLQAS",
            "seed":           self.seed,
            "best_energy":    global_best["energy"],
            "energy_error":   best_err,
            "cnot_count":     global_best["cnot_count"],
            "rotation_count": global_best.get("rotation_count", 0),
            "circuit_depth":  global_best.get("depth", (global_best["step"] + 1) if global_best["step"] is not None else -1),
            "nfev":           -1,
            "success":        int(best_err < accept_err),
            "energy_history": energy_history,
            "cnot_history":   cnot_history,
            "exact_energy":   self.exact_energy,
            "accept_err":     accept_err,
        }

        if self.artifacts is not None:
            self.artifacts.finalize_run_meta(
                wall_clock_sec=time.perf_counter() - run_start,
                final_result={
                    "best_energy_ha": global_best["energy"],
                    "best_energy_error_ha": best_err,
                    "best_cnot_count": global_best["cnot_count"],
                    "best_rotation_count": global_best.get("rotation_count", 0),
                    "best_depth": global_best.get("depth", 0),
                    "success": int(best_err < accept_err),
                },
            )

        out = self.save_result(result)
        print(f"[CRLQASRunner] seed={self.seed}  error={best_err*1000:.3f} mHa  "
              f"cnots={global_best['cnot_count']}  success={bool(result['success'])}  "
              f"→ {out}")
        return result
