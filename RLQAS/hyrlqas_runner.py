"""
HyRLQAS runner for PSQASBench.
Supports both Hybrid_REINFORCE and RENEW variants via config.
"""

from pathlib import Path
import time

import numpy as np
import torch
import wandb

from .utils import get_config, set_seed
from .hy_environment import HyCircuitEnv
from .agents.Hybrid_REINFORCE import HybridActionPolicy
from .agents.Hybrid_REINFORCE_RENEW import HybridActionPolicywithRefine
from .base_runner import BaseRunner


class HyRLQASRunner(BaseRunner):
    """
    Run HyRLQAS (Hybrid Action REINFORCE / RENEW) on one molecule / seed.

    Args:
        config_path : Path to .cfg in PSQASBench/configs/hyrlqas/
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

        self._agent = None   # set in run(), used by greedy/stochastic_episode

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

    def _is_renew(self) -> bool:
        return "RENEW" in self.conf["agent"].get("agent_type", "")

    def _modify_state(self, state: torch.Tensor, env: HyCircuitEnv) -> torch.Tensor:
        if self.conf["agent"].get("en_state"):
            state = torch.cat((
                state,
                torch.tensor([float(env.prev_energy)], dtype=torch.float32,
                             device=self.device),
            ))
        if self.conf["agent"].get("threshold_in_state"):
            state = torch.cat((
                state,
                torch.tensor([float(env.done_threshold)], dtype=torch.float32,
                             device=self.device),
            ))
        return state

    def _save_checkpoint(self, agent, tag: str):
        torch.save(agent.state_dict(), self.result_dir / f"{tag}_model.pth")
        torch.save(agent.optim.state_dict(), self.result_dir / f"{tag}_optim.pth")

    def _make_agent(self, env: HyCircuitEnv):
        conf   = self.conf
        device = self.device
        if self._is_renew():
            base = HybridActionPolicy(conf, env.action_size, env.state_size, device)
            return HybridActionPolicywithRefine(base, conf)
        else:
            return HybridActionPolicy(conf, env.action_size, env.state_size, device)

    # ── BaseRunner: rollouts ──────────────────────────────────────────────────

    def greedy_episode(self, env: HyCircuitEnv) -> dict:
        """Deterministic rollout (argmax discrete, μ continuous)."""
        agent      = self._agent
        accept_err = self.conf["env"]["accept_err"]
        is_renew   = self._is_renew()

        if is_renew:
            agent.HybridAction_policy_net.eval()
            agent.refine_backbone.eval()
        else:
            agent.HybridAction_policy_net.eval()

        with torch.no_grad():
            state = env.reset()
            state = self._modify_state(state, env)
            itr   = 0
            for itr in range(env.num_layers + 1):
                ill = env.illegal_action_new()
                if is_renew:
                    dm = torch.tensor(env.delta_mask, device=agent.device,
                                      dtype=torch.float32)
                    a, act_param, delta_vec = agent.act_with_refine_eval(state, ill, dm)
                    next_state, _, done, _ = env.step(
                        agent.translate[a], act_param,
                        refine_delta=delta_vec, train_flag=False,
                    )
                else:
                    a, act_param = agent.act_eval(state, ill)
                    next_state, _, done, _ = env.step(
                        agent.translate[a], act_param, train_flag=False
                    )
                state = self._modify_state(next_state, env)
                if done:
                    break

        agent.HybridAction_policy_net.train()
        if is_renew:
            agent.refine_backbone.train()

        return {
            "energy_error": float(env.error),
            "cnot_count":   int(env.current_number_of_cnots),
            "success":      int(env.error < accept_err),
            "steps":        itr + 1,
            "op_history":   list(env.op_history),
            "n_qubits":     env.num_qubits,
            "final_state":  env.state.clone(),
        }

    def stochastic_episode(self, env: HyCircuitEnv, seed: int) -> dict:
        """Stochastic rollout with fixed seed (for PCD computation)."""
        agent      = self._agent
        accept_err = self.conf["env"]["accept_err"]
        is_renew   = self._is_renew()

        np.random.seed(seed)
        torch.manual_seed(seed)

        agent.HybridAction_policy_net.eval()

        with torch.no_grad():
            state = env.reset()
            state = self._modify_state(state, env)
            itr   = 0
            for itr in range(env.num_layers + 1):
                ill = env.illegal_action_new()
                if is_renew:
                    dm = torch.tensor(env.delta_mask, device=agent.device,
                                      dtype=torch.float32)
                    a, act_param, _, _, delta_vec, _ = agent.act_with_refine(
                        state, ill, dm
                    )
                    next_state, _, done, _ = env.step(
                        agent.translate[a], act_param,
                        refine_delta=delta_vec.detach(), train_flag=False,
                    )
                else:
                    a, act_param, _, _ = agent.act(state, ill)
                    next_state, _, done, _ = env.step(
                        agent.translate[a], act_param, train_flag=False
                    )
                state = self._modify_state(next_state, env)
                if done:
                    break

        agent.HybridAction_policy_net.train()

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

    def _one_episode(self, ep: int, env: HyCircuitEnv, agent,
                     global_best: dict, energy_history: list, cnot_history: list):
        is_renew = self._is_renew()
        state    = env.reset()
        state    = self._modify_state(state, env)
        agent.HybridAction_policy_net.train()
        traj      = []
        ep_return = 0.0

        for itr in range(env.num_layers + 1):
            ill = env.illegal_action_new()

            if is_renew:
                dm = torch.tensor(env.delta_mask, device=agent.device, dtype=torch.float32)
                a, act_param, logp_type, logp_param, delta_sample, logp_delta = \
                    agent.act_with_refine(state, ill, dm)
                next_state, reward, done, _ = env.step(
                    agent.translate[a], act_param,
                    refine_delta=delta_sample.detach(),
                )
                step_record = {
                    "state":       state.detach().clone().to(agent.device),
                    "action":      a,
                    "act_param":   (None if act_param is None else float(act_param)),
                    "reward":      float(reward),
                    "ill_action":  list(ill) if ill else [],
                    "delta_mask":  dm.detach().cpu().numpy().tolist(),
                    "delta_sample": delta_sample.detach().cpu().numpy().tolist(),
                }
            else:
                a, act_param, _, _ = agent.act(state, ill)
                next_state, reward, done, _ = env.step(agent.translate[a], act_param)
                step_record = {
                    "state":      state.detach().clone().to(agent.device),
                    "action":     a,
                    "act_param":  (None if act_param is None else float(act_param)),
                    "reward":     float(reward),
                    "ill_action": list(ill) if ill else [],
                }

            traj.append(step_record)
            state      = self._modify_state(next_state, env)
            ep_return += float(reward)

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

            if done:
                break

        energy_history.append(float(env.energy))
        cnot_history.append(int(env.current_number_of_cnots))
        return traj, ep_return

    def _train(self, env: HyCircuitEnv, agent, episodes: int):
        mol_name   = Path(self.mol_path).stem
        config_tag = self.config_path.stem
        is_renew   = self._is_renew()
        batch_size = agent.batch_size
        eval_every = self.conf["general"].get("eval_every", 1000)
        eval_K     = self.conf["general"].get("eval_K", 20)
        log_every  = max(1, int(self.conf["general"].get("log_every", 10)))
        save_every = max(1, int(self.conf["general"].get("save_every", 200)))
        method_tag = "hyrlqas"
        _ham_type  = self.conf["problem"].get("ham_type", "mol")
        _nq        = self.conf["env"]["num_qubits"]
        _mol_short = f"{_ham_type}_{_nq}q"
        _optim     = self.conf.get("non_local_opt", {}).get("optim_alg", "noopt")
        _device    = self.device.type
        _run_name  = f"{method_tag}_{_mol_short}_{_optim}_{_device}_seed{self.seed}"
        _group     = f"{method_tag}_{_mol_short}_{_optim}_{_device}"

        wandb.init(
            project="PSQASBench",
            entity="jiayangniu14-rmit-university",
            name=_run_name,
            group=_group,
            config={**self.conf, "config_name": self.config_path.name},
        )
        wandb.define_metric("episode")
        wandb.define_metric("*", step_metric="episode")

        global_best    = {"energy": float("inf"), "episode": None, "step": None,
                          "error": None, "cnot_count": 0, "op_history": None,
                          "moments": None, "state": None}
        energy_history = []
        cnot_history   = []
        batch_trajs    = []
        t_train_start  = time.perf_counter()

        for ep in range(episodes):
            t_ep = time.perf_counter()
            traj, ep_return = self._one_episode(
                ep, env, agent, global_best, energy_history, cnot_history
            )
            batch_trajs.append(traj)
            ep_ms = (time.perf_counter() - t_ep) * 1000.0
            elapsed = time.perf_counter() - t_train_start

            if len(batch_trajs) >= batch_size:
                if is_renew:
                    loss = agent.gradient_update_batch_refine(
                        batch_trajs, agent.entropy_coef, agent.grad_clip
                    )
                else:
                    loss = agent.gradient_update_batch(
                        batch_trajs, agent.entropy_coef, agent.grad_clip
                    )
                wandb.log({"train/policy_loss": loss, "episode": ep})
                batch_trajs = []

            wandb.log({
                "train/episode_energy":    energy_history[-1],
                "train/reward_accumulate": ep_return,
                "episode": ep,
            })

            if ep % log_every == 0:
                err_mha = abs(global_best["energy"] - env.min_energy) * 1000.0
                print(
                    f"[ep {ep:5d}/{episodes}] "
                    f"t={ep_ms:6.0f}ms  "
                    f"energy={energy_history[-1]:.5f}  "
                    f"best={global_best['energy']:.5f}  "
                    f"err={err_mha:.2f}mHa  "
                    f"cnots={global_best['cnot_count']}  "
                    f"elapsed={elapsed:.0f}s",
                    flush=True,
                )

            # periodic evaluation
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

            # periodic save
            if ep % save_every == 0 and ep > 0 and global_best["state"] is not None:
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

    def _train_parallel(self, envs: list[HyCircuitEnv], agent, total_episodes: int):
        """Parallel multi-env trainer with grouped batched VQE kernels."""
        K         = len(envs)
        is_renew  = self._is_renew()
        batch_size = int(agent.batch_size)
        eval_every = self.conf["general"].get("eval_every", 1000)
        eval_K     = self.conf["general"].get("eval_K", 20)
        log_every  = max(1, int(self.conf["general"].get("log_every", 10)))
        save_every = max(1, int(self.conf["general"].get("save_every", 200)))
        method_tag = "hyrlqas"
        mol_name   = Path(self.mol_path).stem
        config_tag = self.config_path.stem
        _ham_type  = self.conf["problem"].get("ham_type", "mol")
        _nq        = self.conf["env"]["num_qubits"]
        _mol_short = f"{_ham_type}_{_nq}q"
        _optim     = self.conf.get("non_local_opt", {}).get("optim_alg", "noopt")
        _device    = self.device.type
        _run_name  = f"{method_tag}_{_mol_short}_{_optim}_{_device}_seed{self.seed}"
        _group     = f"{method_tag}_{_mol_short}_{_optim}_{_device}"

        wandb.init(
            project="PSQASBench",
            entity="jiayangniu14-rmit-university",
            name=_run_name,
            group=_group,
            config={**self.conf, "num_parallel_envs": K, "config_name": self.config_path.name},
        )
        wandb.define_metric("episode")
        wandb.define_metric("*", step_metric="episode")

        # Dedicated eval env — periodic_eval runs K greedy episodes that modify
        # the env's moments/step_counter.  Using a training env would corrupt its
        # circuit state between rounds (greedy_rollout_k only saves curriculum).
        _eval_env = HyCircuitEnv(self.conf, device=self.device, use_gpu_state=False,
                                 shared_bvqe=envs[0]._batched_vqe)

        global_best    = {"energy": float("inf"), "episode": None, "step": None,
                          "error": None, "cnot_count": 0, "op_history": None,
                          "moments": None, "state": None}
        energy_history = []
        cnot_history   = []
        batch_trajs    = []

        states      = [None] * K
        trajs       = [None] * K
        ep_returns  = [0.0] * K
        ep_starts   = [0.0] * K
        active      = [False] * K

        episodes_started = 0
        episodes_done    = 0
        t_train_start    = time.perf_counter()
        cfg_non_local = self.conf.get("non_local_opt", {})
        use_global_batched_rotosolve = self._as_bool(cfg_non_local.get("global_batched_rotosolve", 1), default=True) and all(
            (env.optim_method == "scipy_each_step" and env.optim_alg == "Rotosolve" and not env.phys_noise)
            for env in envs
        )
        use_global_batched_spsa = self._as_bool(cfg_non_local.get("global_batched_spsa", 1), default=True) and all(
            (env.optim_method == "scipy_each_step" and str(env.optim_alg).upper() in {"SPSA", "ADAMSPSA"} and not env.phys_noise)
            for env in envs
        )
        use_global_batched_psr = self._as_bool(cfg_non_local.get("global_batched_psr", 1), default=True) and all(
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
            print(
                f"[parallel-opt] mode={optimizer_mode}  envs={K}  "
                f"global_iters={int(envs[0].global_iters)}",
                flush=True,
            )
        round_idx = 0
        last_progress_log = t_train_start

        def _start_env(k: int):
            nonlocal episodes_started
            if episodes_started >= total_episodes:
                active[k] = False
                return
            st = envs[k].reset()
            states[k] = self._modify_state(st, envs[k])
            trajs[k] = []
            ep_returns[k] = 0.0
            ep_starts[k] = time.perf_counter()
            active[k] = True
            episodes_started += 1

        for k in range(K):
            _start_env(k)

        while episodes_done < total_episodes:
            round_idx += 1
            agent.HybridAction_policy_net.train()
            if is_renew:
                agent.refine_backbone.train()

            active_idxs = [k for k in range(K) if active[k]]
            if not active_idxs:
                break

            step_meta = {}
            step_states = {}
            opt_ref_states = {}

            # 1) sample actions per active env
            for k in active_idxs:
                env = envs[k]
                state = states[k]
                ill = env.illegal_action_new()

                if is_renew:
                    dm = torch.tensor(env.delta_mask, device=agent.device, dtype=torch.float32)
                    a, act_param, _lt, _lp, delta_sample, _ld = agent.act_with_refine(state, ill, dm)
                    meta = {
                        "action_id": a,
                        "action": agent.translate[a],
                        "act_param": act_param,
                        "refine_delta": delta_sample.detach(),
                        "step_record": {
                            "state":        state.detach().clone().to(agent.device),
                            "action":       a,
                            "act_param":    (None if act_param is None else float(act_param)),
                            "reward":       None,
                            "ill_action":   list(ill) if ill else [],
                            "delta_mask":   dm.detach().cpu().numpy().tolist(),
                            "delta_sample": delta_sample.detach().cpu().numpy().tolist(),
                        },
                    }
                else:
                    a, act_param, _lt, _lp = agent.act(state, ill)
                    meta = {
                        "action_id": a,
                        "action": agent.translate[a],
                        "act_param": act_param,
                        "refine_delta": None,
                        "step_record": {
                            "state":      state.detach().clone().to(agent.device),
                            "action":     a,
                            "act_param":  (None if act_param is None else float(act_param)),
                            "reward":     None,
                            "ill_action": list(ill) if ill else [],
                        },
                    }
                step_meta[k] = meta

            # 2) apply actions and optionally defer rotosolve globally
            for k in active_idxs:
                env = envs[k]
                m = step_meta[k]
                step_states[k] = env.step_deferred_energy(
                    m["action"],
                    act_param=m["act_param"],
                    refine_delta=m["refine_delta"],
                    run_optimizer=not use_global_batched_optimizer,
                )
                # Capture post-action structure for grouped batched optimisers.
                opt_ref_states[k] = env.state.clone()

            if use_global_batched_rotosolve:
                active_envs = [envs[k] for k in active_idxs]
                active_opt_states = [opt_ref_states[k] for k in active_idxs]
                self._rotosolve_envs_batched(active_envs, opt_states=active_opt_states)
                for k in active_idxs:
                    step_states[k] = envs[k].state.clone()
            elif use_global_batched_spsa:
                active_envs = [envs[k] for k in active_idxs]
                active_opt_states = [opt_ref_states[k] for k in active_idxs]
                self._spsa_envs_batched(
                    active_envs,
                    opt_states=active_opt_states,
                    method=active_envs[0].optim_alg,
                )
                for k in active_idxs:
                    step_states[k] = envs[k].state.clone()
            elif use_global_batched_psr:
                active_envs = [envs[k] for k in active_idxs]
                active_opt_states = [opt_ref_states[k] for k in active_idxs]
                self._psr_adam_envs_batched(active_envs, opt_states=active_opt_states)
                for k in active_idxs:
                    step_states[k] = envs[k].state.clone()

            # 3) grouped batched energy eval for active envs
            active_envs = [envs[k] for k in active_idxs]
            energies = self._eval_env_energies_batched(active_envs)

            # 4) finalize transitions and handle episode boundaries
            for local_i, k in enumerate(active_idxs):
                env = envs[k]
                m = step_meta[k]
                e, e0 = energies[local_i]
                ns, reward, done, _ = env.finalize_step_with_energy(
                    action=m["action"],
                    next_state=step_states[k],
                    energy=e,
                    energy_noiseless=e0,
                )
                ns_mod = self._modify_state(ns, env)
                states[k] = ns_mod

                m["step_record"]["reward"] = float(reward)
                trajs[k].append(m["step_record"])
                ep_returns[k] += float(reward)

                energy_now = float(env.energy)
                if energy_now < global_best["energy"]:
                    global_best.update({
                        "energy":     energy_now,
                        "episode":    episodes_done,
                        "step":       env.step_counter,
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
                        "episode": episodes_done,
                    })

                if not done:
                    continue

                episode_idx = episodes_done
                episodes_done += 1

                ep_ms = (time.perf_counter() - ep_starts[k]) * 1000.0
                elapsed = time.perf_counter() - t_train_start
                energy_history.append(float(env.energy))
                cnot_history.append(int(env.current_number_of_cnots))
                batch_trajs.append(trajs[k])

                if len(batch_trajs) >= batch_size:
                    if is_renew:
                        loss = agent.gradient_update_batch_refine(
                            batch_trajs, agent.entropy_coef, agent.grad_clip
                        )
                    else:
                        loss = agent.gradient_update_batch(
                            batch_trajs, agent.entropy_coef, agent.grad_clip
                        )
                    wandb.log({"train/policy_loss": loss, "episode": episode_idx})
                    batch_trajs = []

                wandb.log({
                    "train/episode_energy":    energy_history[-1],
                    "train/reward_accumulate": ep_returns[k],
                    "episode": episode_idx,
                })

                if episode_idx % log_every == 0:
                    err_mha = abs(global_best["energy"] - env.min_energy) * 1000.0
                    print(
                        f"[ep {episode_idx:5d}/{total_episodes}  env{k}] "
                        f"energy={energy_history[-1]:.5f}  "
                        f"best={global_best['energy']:.5f}  "
                        f"err={err_mha:.2f}mHa  "
                        f"cnots={global_best['cnot_count']}  "
                        f"elapsed={elapsed:.0f}s",
                        flush=True,
                    )

                if eval_every > 0 and episode_idx > 0 and episode_idx % eval_every == 0:
                    eval_metrics = self.periodic_eval(episode_idx, _eval_env, K=eval_K)
                    wandb.log({
                        "eval/sr_at_chem":     eval_metrics["sr_at_chem"],
                        "eval/cnot_at_chem":   eval_metrics["cnot_at_chem"],
                        "eval/best_error_mha": eval_metrics["best_error_mha"],
                        "eval/mean_error_mha": eval_metrics["mean_error_mha"],
                        "eval/mean_cnots":     eval_metrics["mean_cnots"],
                        "eval/D_struct":       eval_metrics["D_struct"],
                        "eval/D_func":         eval_metrics["D_func"],
                        "episode": episode_idx,
                    })
                    print(
                        f"[eval ep={episode_idx}]  SR={eval_metrics['sr_at_chem']:.2f}  "
                        f"best_err={eval_metrics['best_error_mha']:.2f} mHa  "
                        f"CNOT@chem={eval_metrics['cnot_at_chem']}  "
                        f"D_struct={eval_metrics['D_struct']:.4f}  "
                        f"D_func={eval_metrics['D_func']:.4f}"
                    )

                if episode_idx % save_every == 0 and episode_idx > 0 and global_best["state"] is not None:
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

                _start_env(k)

            elapsed = time.perf_counter() - t_train_start
            if (
                optimizer_mode is not None
                and elapsed - (last_progress_log - t_train_start) >= 60.0
            ):
                active_depths = [int(envs[k].step_counter) for k in active_idxs]
                if active_depths:
                    min_depth = min(active_depths)
                    max_depth = max(active_depths)
                else:
                    min_depth = max_depth = 0
                print(
                    f"[parallel-opt] mode={optimizer_mode}  rounds={round_idx}  "
                    f"episodes_done={episodes_done}/{total_episodes}  "
                    f"active_env_depth_range={min_depth}-{max_depth}  "
                    f"elapsed={elapsed:.0f}s",
                    flush=True,
                )
                last_progress_log = time.perf_counter()

        wandb.finish()
        return global_best, energy_history, cnot_history

    # ── public interface ──────────────────────────────────────────────────────

    def run(self) -> dict:
        set_seed(self.seed)
        torch.set_num_threads(1)

        env   = HyCircuitEnv(self.conf, device=self.device)
        agent = self._make_agent(env)
        self._agent = agent

        _shared_bvqe = env._batched_vqe
        global_best, energy_history, cnot_history = self.train_with_parallel_entry(
            env=env,
            agent=agent,
            episodes=self.conf["general"]["episodes"],
            train_single_fn=self._train,
            make_env_fn=lambda: HyCircuitEnv(self.conf, device=self.device, use_gpu_state=False,
                                             shared_bvqe=_shared_bvqe),
            train_parallel_fn=self._train_parallel,
            require_batch_divisible=False,
        )

        accept_err = self.conf["env"]["accept_err"]
        best_err   = abs(global_best["error"]) if global_best["error"] is not None else float("inf")
        method_name = "RENEW" if self._is_renew() else "Hybrid_REINFORCE"

        result = {
            "method":         method_name,
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
        print(f"[HyRLQASRunner/{method_name}] seed={self.seed}  "
              f"error={best_err*1000:.3f} mHa  cnots={global_best['cnot_count']}  "
              f"success={bool(result['success'])}  → {out}")
        return result
