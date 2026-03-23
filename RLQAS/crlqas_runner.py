"""
CRLQAS runner for PSQASBench.
All code is self-contained within the PSQASBench package — no sys.path tricks.
"""

from pathlib import Path

import numpy as np
import torch

from .utils import get_config, set_seed
from .environment import CircuitEnv
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

        # Parse config using PSQASBench's own get_config
        self.conf = get_config(str(self.config_path))

        # Inject absolute mol_data path so CircuitEnv can find the .npz
        self.conf["problem"]["mol_data_dir"] = str(Path(mol_path).parent)
        self.conf["problem"]["mol_file"]     = Path(mol_path).name

        super().__init__(self.conf, mol_path, result_dir, seed)

    # ── helpers ───────────────────────────────────────────────────────────────

    def _modify_state(self, state, env):
        if self.conf["agent"].get("en_state"):
            state = torch.cat(
                (state, torch.tensor([env.prev_energy], dtype=torch.float,
                                     device=self.device))
            )
        return state

    def _save_checkpoint(self, agent, tag: str):
        torch.save(agent.policy_net.state_dict(), self.result_dir / f"{tag}_model.pth")
        torch.save(agent.optim.state_dict(),       self.result_dir / f"{tag}_optim.pth")

    # ── training loop ─────────────────────────────────────────────────────────

    def _one_episode(self, ep, env, agent, global_best,
                     energy_history, cnot_history):
        agent.saver.get_new_episode("train", ep)
        state = env.reset()
        state = self._modify_state(state, env)
        agent.policy_net.train()

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
            agent.saver.stats_file["train"][ep]["errors"].append(env.error)
            agent.saver.stats_file["train"][ep]["errors_noiseless"].append(env.error_noiseless)

            energy_now = float(env.energy)
            if energy_now < global_best["energy"]:
                global_best.update({
                    "energy":     energy_now,
                    "episode":    ep,
                    "step":       itr,
                    "error":      energy_now - env.min_energy,
                    "cnot_count": int(env.current_number_of_cnots),
                    "op_history": list(env.op_history),
                    "moments":    list(env.moments),
                    "state":      env.state.clone(),
                })
                self._save_checkpoint(
                    agent,
                    tag=f"best_thresh{self.conf['env']['accept_err']}_seed{self.seed}",
                )

            if len(agent.memory) > self.conf["agent"]["batch_size"]:
                agent.replay(self.conf["agent"]["batch_size"])

            if done:
                break

        energy_history.append(float(env.energy))
        cnot_history.append(int(env.current_number_of_cnots))

    def _train(self, env, agent, episodes):
        global_best    = {"energy": float("inf"), "episode": None, "step": None,
                          "error": None, "cnot_count": 0, "op_history": None,
                          "moments": None, "state": None}
        energy_history = []
        cnot_history   = []

        for ep in range(episodes):
            self._one_episode(ep, env, agent, global_best,
                              energy_history, cnot_history)
            if ep % 20 == 0 and ep > 0:
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
        agent.saver = _Saver(self.result_dir, self.seed)

        global_best, energy_history, cnot_history = self._train(
            env, agent, self.conf["general"]["episodes"]
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
