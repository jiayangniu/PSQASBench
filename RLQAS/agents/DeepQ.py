"""
DQN base agent for PSQASBench.
Adapted from CRLQAS/agents/DeepQ.py — only import path changed.
"""

import copy
import random
from collections import namedtuple, deque

import numpy as np
import torch
import torch.nn as nn

from ..utils import get_action_dict, dict_of_actions_revert_q


class DQN:

    def __init__(self, conf, action_size, state_size, device):
        self.num_qubits    = conf["env"]["num_qubits"]
        self.num_layers    = conf["env"]["num_layers"]
        self.batch_size    = int(conf["agent"]["batch_size"])
        self.final_gamma   = conf["agent"]["final_gamma"]
        self.epsilon_min   = conf["agent"]["epsilon_min"]
        self.epsilon_decay = conf["agent"]["epsilon_decay"]
        lr                 = conf["agent"]["learning_rate"]
        self.update_target_net = conf["agent"]["update_target_net"]
        neuron_list        = conf["agent"]["neurons"]
        drop_prob          = conf["agent"]["dropout"]
        self.with_angles   = conf["agent"]["angles"]

        mem_reset = conf["agent"].get("memory_reset_switch", False)
        self.memory_reset_switch     = mem_reset
        self.memory_reset_threshold  = conf["agent"].get("memory_reset_threshold", False)
        self.memory_reset_counter    = 0

        self.action_size = action_size
        self.state_size  = state_size if self.with_angles else state_size - self.num_layers * self.num_qubits * 3
        if conf["agent"].get("en_state"):
            self.state_size += 1
        if conf["agent"].get("threshold_in_state"):
            self.state_size += 1

        connectivity = conf["env"].get("connectivity", "all")
        self.translate     = get_action_dict(self.num_qubits, connectivity)
        self.rev_translate = dict_of_actions_revert_q(self.num_qubits)
        self.policy_net    = self._build_network(neuron_list, drop_prob).to(device)
        self.target_net    = copy.deepcopy(self.policy_net)
        self.target_net.eval()

        self.gamma         = torch.tensor(
            [np.round(np.power(self.final_gamma, 1 / self.num_layers), 2)]
        ).to(device)
        self.memory        = ReplayMemory(conf["agent"]["memory_size"])
        self.epsilon       = 1.0
        self.optim         = torch.optim.Adam(self.policy_net.parameters(), lr=lr)
        self.loss          = nn.SmoothL1Loss()
        self.device        = device
        self.step_counter  = 0

        self.Transition    = namedtuple(
            "Transition", ("state", "action", "reward", "next_state", "done")
        )

        # Optional importance-weighted exploration (activated by qubit_importance in config).
        imp_raw = conf["agent"].get("qubit_importance")
        if imp_raw is not None:
            imp = np.array(imp_raw, dtype=float)
            N   = self.num_qubits
            w   = np.empty(action_size)
            for idx, v in self.translate.items():
                if v[0] < N:   # CNOT: [ctrl, offset, N, 0]
                    targ  = (v[0] + v[1]) % N
                    w[idx] = (imp[v[0]] + imp[targ]) / 2.0
                else:           # Rotation: [N, 0, rot_qubit, rot_axis]
                    w[idx] = imp[v[2]]
            self.explore_weights = (w / w.sum()).astype(np.float64)
        else:
            self.explore_weights = None

    def _build_network(self, neurons, p):
        layers = []
        sizes  = [self.state_size] + neurons
        for in_n, out_n in zip(sizes[:-1], sizes[1:]):
            layers += [nn.Linear(in_n, out_n), nn.LeakyReLU(), nn.Dropout(p=p)]
        layers.append(nn.Linear(sizes[-1], self.action_size))
        return nn.Sequential(*layers)

    def remember(self, state, action, reward, next_state, done):
        self.memory.push(state, action, reward, next_state, done)

    def _explore(self, ill_actions) -> int:
        """Sample a random legal action, optionally weighted by qubit importance."""
        if self.explore_weights is not None:
            ill_set = set(ill_actions)
            legal   = [a for a in range(self.action_size) if a not in ill_set]
            w       = self.explore_weights[legal]
            return int(np.random.choice(legal, p=w / w.sum()))
        a = torch.randint(self.action_size, (1,)).item()
        while a in ill_actions:
            a = torch.randint(self.action_size, (1,)).item()
        return a

    def act(self, state, ill_actions):
        if torch.rand(1).item() <= self.epsilon:
            return self._explore(ill_actions), True
        q = self.policy_net(state.unsqueeze(0))
        q[0][list(ill_actions)] = float("-inf")
        return torch.argmax(q[0]).item(), False

    def act_batch(self, states_list, ill_actions_list):
        """Batch ε-greedy for K envs — one GPU forward pass instead of K separate calls.

        Args:
            states_list      : list of K state tensors (each already on self.device)
            ill_actions_list : list of K illegal-action index lists

        Returns:
            list of K (action_int, is_random) tuples, same schema as act().
        """
        K     = len(states_list)
        batch = torch.stack(states_list)          # (K, state_size) on GPU
        with torch.no_grad():
            q_batch = self.policy_net(batch)      # (K, action_size)

        results = []
        for k in range(K):
            if torch.rand(1).item() <= self.epsilon:
                results.append((self._explore(ill_actions_list[k]), True))
            else:
                q = q_batch[k].clone()
                q[list(ill_actions_list[k])] = float("-inf")
                results.append((torch.argmax(q).item(), False))
        return results

    def replay(self, batch_size):
        if self.step_counter % self.update_target_net == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        self.step_counter += 1

        transitions = self.memory.sample(batch_size)
        batch       = self.Transition(*zip(*transitions))

        s   = torch.stack(batch.state)
        a   = torch.stack(batch.action)
        r   = torch.stack(batch.reward)
        ns  = torch.stack(batch.next_state)
        d   = torch.stack(batch.done)

        q_sa  = self.policy_net(s).gather(1, a.unsqueeze(1))
        # Double DQN
        next_a = self.policy_net(ns).max(1)[1].detach()
        next_q = self.target_net(ns).gather(1, next_a.unsqueeze(1)).squeeze(1)
        target = (next_q * self.gamma * (1 - d) + r).view(-1, 1)

        self.optim.zero_grad()
        loss = self.loss(q_sa, target)
        loss.backward()
        self.optim.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)
        return loss.item()

    # alias so DQN_Nstep can call super().replay via fit
    def fit(self, output, target):
        self.optim.zero_grad()
        l = self.loss(output, target)
        l.backward()
        self.optim.step()
        return l.item()


class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory   = []
        self.position = 0
        self.Transition = namedtuple(
            "Transition", ("state", "action", "reward", "next_state", "done")
        )

    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = self.Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

    def clean_memory(self):
        self.memory   = []
        self.position = 0
