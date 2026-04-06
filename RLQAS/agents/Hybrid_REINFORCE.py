"""
Hybrid Action REINFORCE policy for PSQASBench.
Adapted from HyRLQAS/agents/Hybrid_REINFORCE.py.
Changes: package-relative imports only (no sys.path tricks).
"""
from types import MethodType

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Normal

from ..utils import get_action_dict, dict_of_actions_revert_q


class HybridActionPolicy(nn.Module):

    def __init__(self, conf, action_size, state_size, device):
        super().__init__()
        self.device     = device
        self.num_qubits = conf["env"]["num_qubits"]
        self.num_layers = conf["env"]["num_layers"]
        self.batch_size = conf["agent"]["batch_size"]

        self.final_gamma      = conf["agent"]["final_gamma"]
        self.gamma            = torch.tensor(
            [np.round(np.power(self.final_gamma, 1 / self.num_layers), 2)],
            dtype=torch.float32,
        ).to(device)
        self.normalize_returns = conf["agent"]["normalize_returns"]
        self.entropy_coef      = float(conf["agent"]["entropy_coef"])
        self.grad_clip         = float(conf["agent"]["grad_clip"])

        self.discrete_action_size   = action_size
        self.continuous_action_size = action_size

        # only rotation actions carry a continuous parameter
        self.continuous_action_mask = torch.zeros(
            self.discrete_action_size, dtype=torch.float32, device=device
        )
        self.continuous_action_mask[-3 * self.num_qubits :] = 1.0

        # HyCircuitEnv can emit either structure-only state (angles=0) or full
        # state with angle channels (angles=1). Keep network input dim aligned.
        self.with_angles = conf["agent"].get("angles", 1)
        self.state_size = state_size if self.with_angles else (
            state_size - self.num_layers * self.num_qubits * 3
        )
        if conf["agent"].get("en_state"):
            self.state_size += 1
        if conf["agent"].get("threshold_in_state"):
            self.state_size += 1

        connectivity = conf["env"].get("connectivity", "all")
        self.translate     = get_action_dict(self.num_qubits, connectivity)
        self.rev_translate = dict_of_actions_revert_q(self.num_qubits)

        neuron_list = conf["agent"]["neurons"]
        drop_prob   = conf["agent"]["dropout"]
        self.HybridAction_policy_net = self._build_network(neuron_list, drop_prob).to(device)
        self.optim = torch.optim.Adam(
            self.HybridAction_policy_net.parameters(),
            lr=conf["agent"]["learning_rate"],
        )

    # ── network construction ──────────────────────────────────────────────────

    def _build_network(self, neuron_list, drop_prob) -> nn.Module:
        layers, in_dim = [], self.state_size
        for h in neuron_list:
            layers += [nn.Linear(in_dim, h), nn.LeakyReLU(), nn.Dropout(p=drop_prob)]
            in_dim = h
        backbone = nn.Sequential(*layers)

        discrete_action_head = nn.Linear(in_dim, self.discrete_action_size)
        params_mu_head       = nn.Linear(in_dim, self.continuous_action_size)
        params_log_sigma     = nn.Parameter(torch.zeros(self.continuous_action_size))

        net = nn.Module()
        net.backbone             = backbone
        net.discrete_action_head = discrete_action_head
        net.params_mu_head       = params_mu_head
        net.params_log_sigma     = params_log_sigma

        def _forward(net, x: torch.Tensor):
            feat   = net.backbone(x)
            logits = net.discrete_action_head(feat)
            mu     = net.params_mu_head(feat)
            sigma  = torch.exp(net.params_log_sigma).unsqueeze(0).expand_as(mu)
            return {"action_type": logits, "action_parms": {"mu": mu, "sigma": sigma}}

        net.forward = MethodType(_forward, net)
        return net

    # ── action selection ──────────────────────────────────────────────────────

    @torch.no_grad()
    def act(self, state: torch.Tensor, ill_action: list, greedy: bool = False):
        state  = state.unsqueeze(0).to(self.device)
        out    = self.HybridAction_policy_net(state)
        logits = out["action_type"].clone()
        mu     = out["action_parms"]["mu"].clone()
        sigma  = out["action_parms"]["sigma"].clone()
        action_size = logits.size(1)

        if ill_action:
            logits[0, torch.tensor(ill_action, dtype=torch.long, device=self.device)] = -1e9

        if greedy:
            discrete_action = logits.argmax(dim=-1)
            logp_type = torch.zeros(1, device=self.device)
        else:
            dist           = Categorical(logits=logits)
            discrete_action = dist.sample()
            logp_type       = dist.log_prob(discrete_action)

        oh      = F.one_hot(discrete_action, num_classes=action_size).float()
        has_mask = oh * self.continuous_action_mask.unsqueeze(0)

        dist_cont        = Normal(mu, sigma)
        all_cont         = dist_cont.sample()
        continuous_action = (all_cont * has_mask).sum(dim=-1)
        logp_param        = (dist_cont.log_prob(all_cont) * has_mask).sum(dim=-1)

        a = int(discrete_action.item())
        act_param = float(continuous_action.item()) if self.continuous_action_mask[a] else None
        return a, act_param, logp_type.squeeze(0), logp_param.squeeze(0)

    @torch.no_grad()
    def act_eval(self, state: torch.Tensor, ill_action: list):
        """Deterministic (argmax discrete, μ continuous)."""
        state = state.unsqueeze(0).to(self.device)
        was_training = self.HybridAction_policy_net.training
        self.HybridAction_policy_net.eval()

        out    = self.HybridAction_policy_net(state)
        logits = out["action_type"].clone()
        mu     = out["action_parms"]["mu"].clone()

        if ill_action:
            logits[0, torch.tensor(ill_action, dtype=torch.long, device=self.device)] = -1e9

        a = int(logits.argmax(dim=-1).item())
        act_param = float(mu[0, a].item()) if self.continuous_action_mask[a] else None

        if was_training:
            self.HybridAction_policy_net.train()
        return a, act_param

    # ── policy update ─────────────────────────────────────────────────────────

    def gradient_update_batch(self, batch_trajs, entropy_coef: float = 0.0,
                              grad_clip: float = 0.0):
        device      = self.device
        step_gamma  = float(self.gamma.item())
        action_size = self.discrete_action_size

        per_ep_G, all_G = [], []
        for traj in batch_trajs:
            rewards = [float(s["reward"]) for s in traj]
            G, running = [], 0.0
            for r in reversed(rewards):
                running = r + step_gamma * running
                G.append(running)
            G.reverse()
            per_ep_G.append(G)
            all_G.extend(G)

        all_G_t = torch.tensor(all_G, dtype=torch.float32, device=device)
        if self.normalize_returns == "global" and all_G_t.numel() > 1:
            all_G_t = (all_G_t - all_G_t.mean()) / (all_G_t.std() + 1e-8)
        elif self.normalize_returns == "per_episode":
            cursor = 0
            for G in per_ep_G:
                T = len(G)
                if T > 1:
                    g = torch.tensor(G, dtype=torch.float32, device=device)
                    all_G_t[cursor: cursor + T] = (g - g.mean()) / (g.std() + 1e-8)
                cursor += T

        losses, entropies, cursor = [], [], 0
        for traj in batch_trajs:
            for step in traj:
                state  = step["state"].unsqueeze(0).to(device)
                a_disc = int(step["action"])
                a_cont = step["act_param"]
                ill    = step.get("ill_action", [])

                out    = self.HybridAction_policy_net(state)
                logits = out["action_type"].clone()
                mu     = out["action_parms"]["mu"]
                sigma  = out["action_parms"]["sigma"]
                if ill:
                    logits[0, torch.as_tensor(ill, device=device, dtype=torch.long)] = -1e9

                a_t        = torch.tensor([a_disc], device=device, dtype=torch.long)
                dist_disc  = Categorical(logits=logits)
                logp_disc  = dist_disc.log_prob(a_t)

                if self.continuous_action_mask[a_disc] == 0 or a_cont is None:
                    logp_cont = torch.zeros(1, device=device)
                else:
                    oh       = F.one_hot(a_t, num_classes=action_size).float()
                    has_mask = oh * self.continuous_action_mask.unsqueeze(0)
                    mu_full  = mu.detach().clone()
                    mu_full[0, a_disc] = float(a_cont)
                    logp_cont = (Normal(mu, sigma).log_prob(mu_full) * has_mask).sum(dim=-1)

                G_t = all_G_t[cursor]
                cursor += 1
                losses.append(-(logp_disc + logp_cont) * G_t)
                if entropy_coef != 0.0:
                    with torch.no_grad():
                        entropies.append(dist_disc.entropy().squeeze(0))

        policy_loss = torch.stack(losses).mean()
        if entropy_coef != 0.0 and entropies:
            policy_loss = policy_loss - entropy_coef * torch.stack(entropies).mean()

        self.optim.zero_grad()
        policy_loss.backward()
        if grad_clip and grad_clip > 0:
            nn.utils.clip_grad_norm_(self.HybridAction_policy_net.parameters(), grad_clip)
        self.optim.step()
        return float(policy_loss.detach().cpu())
