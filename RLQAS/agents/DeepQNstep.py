"""
N-step DQN agent for PSQASBench.
Adapted from CRLQAS/agents/DeepQNstep.py — only import path changed.
"""

import random
from collections import namedtuple, deque

from .DeepQ import DQN


class DQN_Nstep(DQN):

    def __init__(self, conf, action_size, state_size, device):
        super().__init__(conf, action_size, state_size, device)
        self.memory = N_step_ReplayMemory(
            conf["agent"]["memory_size"],
            conf["agent"]["n_step"],
            self.gamma,
        )

    def replay(self, batch_size):
        if self.step_counter % self.update_target_net == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        self.step_counter += 1

        import torch
        transitions = self.memory.sample(batch_size)
        batch       = self.Transition(*zip(*transitions))

        s   = torch.stack(batch.state)
        a   = torch.stack(batch.action)
        r   = torch.stack(batch.reward)
        ns  = torch.stack(batch.next_state)
        d   = torch.stack(batch.done)

        q_sa   = self.policy_net(s).gather(1, a.unsqueeze(1))
        next_a = self.policy_net(ns).max(1)[1].detach()
        next_q = self.target_net(ns).gather(1, next_a.unsqueeze(1)).squeeze(1)
        target = (next_q * self.gamma * (1 - d) + r).view(-1, 1)

        cost = self.fit(q_sa, target)
        if self.epsilon > self.epsilon_min:
            self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)
        return cost


class N_step_ReplayMemory:

    def __init__(self, capacity, n_step, gamma):
        self.capacity       = capacity
        self.n_step         = n_step
        self.gamma          = gamma
        self.memory         = deque(maxlen=capacity)
        self.n_step_memory  = deque(maxlen=n_step)
        self.Transition     = namedtuple(
            "Transition", ("state", "action", "reward", "next_state", "done")
        )

    def _n_step_return(self):
        reward, n_state, done = self.n_step_memory[-1][-3:]
        for _, _, rwd, next_st, do in list(self.n_step_memory)[::-1][1:]:
            reward  = self.gamma * reward * (1 - do) + rwd
            n_state, done = (next_st, do) if do else (n_state, done)
        return reward, n_state, done

    def push(self, *args):
        self.n_step_memory.append(self.Transition(*args))
        if len(self.n_step_memory) < self.n_step:
            return
        reward, n_state, done = self._n_step_return()
        state, action         = self.n_step_memory[0][:2]
        # reward can be 0-d tensor (scalar) or length-1 tensor depending on caller.
        # Keep a scalar tensor for replay stacking while avoiding shape-index errors.
        reward_scalar = reward.reshape(-1)[0] if hasattr(reward, "reshape") else reward
        self.memory.append(self.Transition(state, action, reward_scalar, n_state, done))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

    def clean_memory(self):
        self.memory        = deque(maxlen=self.capacity)
        self.n_step_memory = deque(maxlen=self.n_step)
