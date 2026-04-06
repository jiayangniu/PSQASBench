"""
RENEW variant of Hybrid Action REINFORCE for PSQASBench.
Adapted from HyRLQAS/agents/Hybrid_REINFORCE_RENEW.py.
Changes: package-relative imports only.
"""
from types import MethodType

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Normal


def _forward_with_latent(net: nn.Module, x: torch.Tensor):
    feat         = net.backbone(x)
    logits       = net.discrete_action_head(feat)
    mu           = net.params_mu_head(feat)
    sigma        = torch.exp(net.params_log_sigma).unsqueeze(0).expand_as(mu)
    param_latent = net.param_latent_head(feat)
    return {
        "action_type":  logits,
        "action_parms": {"mu": mu, "sigma": sigma},
        "param_latent": param_latent,
    }


class HybridActionPolicywithRefine(nn.Module):
    """
    Wraps HybridActionPolicy and adds a parameter-refine head.
    At each step, the refine head outputs Δθ increments for all prior
    rotation gates, allowing the policy to adjust historical angles.
    """

    def __init__(self, base_agent, conf):
        super().__init__()
        self.base_agent = base_agent
        self.device     = base_agent.device

        # share all base-agent components
        self.HybridAction_policy_net  = base_agent.HybridAction_policy_net
        self.discrete_action_size     = base_agent.discrete_action_size
        self.continuous_action_size   = base_agent.continuous_action_size
        self.continuous_action_mask   = base_agent.continuous_action_mask
        self.batch_size               = base_agent.batch_size
        self.translate                = base_agent.translate
        self.state_size               = base_agent.state_size
        self.gamma                    = base_agent.gamma
        self.normalize_returns        = base_agent.normalize_returns
        self.entropy_coef             = base_agent.entropy_coef
        self.grad_clip                = base_agent.grad_clip
        self.optim                    = base_agent.optim

        self.param_latent_dim  = int(conf["agent"].get("param_latent_dim", 256))
        self.action_emb_dim    = int(conf["agent"].get("action_emb_dim", 32))
        self.max_params        = base_agent.num_layers
        self.delta_max         = float(conf["agent"].get("delta_max", 0.0))
        self.refine_entropy_coef = float(conf["agent"].get("refine_entropy_coef", 0.0))

        # add param_latent_head to backbone and swap forward
        feat_dim = next(
            m.out_features for m in reversed(list(self.HybridAction_policy_net.backbone))
            if isinstance(m, nn.Linear)
        )
        self.HybridAction_policy_net.param_latent_head = nn.Linear(
            feat_dim, self.param_latent_dim
        ).to(self.device)
        self.HybridAction_policy_net.forward = MethodType(
            _forward_with_latent, self.HybridAction_policy_net
        )

        self.action_embedding = nn.Embedding(
            self.discrete_action_size, self.action_emb_dim
        ).to(self.device)

        refine_in_dim = self.state_size + self.action_emb_dim + self.param_latent_dim
        hidden_list   = conf["agent"].get("refine_neurons", [2000, 2000])
        drop_prob     = conf["agent"].get("refine_dropout", 0.0)

        layers, in_dim = [], refine_in_dim
        for h in hidden_list:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.LeakyReLU())
            if drop_prob and drop_prob > 0:
                layers.append(nn.Dropout(p=drop_prob))
            in_dim = h
        self.refine_backbone = nn.Sequential(*layers).to(self.device)
        self.refine_mu_head  = nn.Linear(in_dim, self.max_params).to(self.device)
        self.refine_logsigma = nn.Parameter(
            torch.zeros(self.max_params, device=self.device)
        )

        # register new params with existing optimiser
        self.optim.add_param_group({
            "params": (
                list(self.HybridAction_policy_net.param_latent_head.parameters()) +
                list(self.refine_backbone.parameters()) +
                list(self.refine_mu_head.parameters()) +
                [self.refine_logsigma]
            )
        })

    # ── action selection ──────────────────────────────────────────────────────

    def act_with_refine(self, state: torch.Tensor, ill_action: list,
                        delta_mask: torch.Tensor, greedy: bool = False):
        """Sample discrete + continuous action, then sample Δθ refine vector."""
        device = self.device
        action, act_param, logp_type, logp_param = self.base_agent.act(
            state, ill_action, greedy
        )

        state_2d   = state.unsqueeze(0).to(device)
        out        = self.HybridAction_policy_net(state_2d)
        param_latent = out["param_latent"]

        a_t         = torch.tensor([action], dtype=torch.long, device=device)
        action_emb  = self.action_embedding(a_t)
        refine_in   = torch.cat([state_2d, action_emb, param_latent], dim=-1)
        feat        = self.refine_backbone(refine_in)
        mu_delta    = self.refine_mu_head(feat)
        sigma_delta = torch.exp(self.refine_logsigma).unsqueeze(0).expand_as(mu_delta)

        dist_delta  = Normal(mu_delta, sigma_delta)
        delta_sample = dist_delta.sample()
        delta_mask_2d = delta_mask.to(delta_sample.dtype).unsqueeze(0)
        logp_delta    = (dist_delta.log_prob(delta_sample) * delta_mask_2d).sum(dim=-1)
        delta_sample  = delta_sample * delta_mask_2d

        return (
            int(action),
            act_param,
            logp_type,
            logp_param,
            delta_sample.squeeze(0),
            logp_delta.squeeze(0),
        )

    @torch.no_grad()
    def act_with_refine_eval(self, state: torch.Tensor, ill_action: list,
                             delta_mask: torch.Tensor):
        """Deterministic rollout: argmax discrete, μ continuous, μ_delta refine."""
        device = self.device
        state_2d = state.unsqueeze(0).to(device)

        was_main = self.HybridAction_policy_net.training
        was_ref  = self.refine_backbone.training
        self.HybridAction_policy_net.eval()
        self.refine_backbone.eval()

        out    = self.HybridAction_policy_net(state_2d)
        logits = out["action_type"].clone()
        mu     = out["action_parms"]["mu"].clone()
        param_latent = out["param_latent"]

        if ill_action:
            logits[0, torch.as_tensor(ill_action, device=device, dtype=torch.long)] = -1e9

        a = int(logits.argmax(dim=-1).item())
        act_param = float(mu[0, a].item()) if self.continuous_action_mask[a] else None

        a_t        = torch.tensor([a], dtype=torch.long, device=device)
        action_emb = self.action_embedding(a_t)
        refine_in  = torch.cat([state_2d, action_emb, param_latent], dim=-1)
        feat       = self.refine_backbone(refine_in)
        mu_delta   = self.refine_mu_head(feat).squeeze(0)

        delta_mask_d = delta_mask.to(device=device, dtype=mu_delta.dtype)
        delta_vec    = mu_delta * delta_mask_d
        if self.delta_max > 0:
            delta_vec = torch.clamp(delta_vec, -self.delta_max, self.delta_max)

        if was_main:
            self.HybridAction_policy_net.train()
        if was_ref:
            self.refine_backbone.train()

        return a, act_param, delta_vec.detach()

    # ── policy update ─────────────────────────────────────────────────────────

    def gradient_update_batch_refine(self, batch_trajs, entropy_coef: float = 0.0,
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
                state_2d = step["state"].unsqueeze(0).to(device)
                out      = self.HybridAction_policy_net(state_2d)
                logits   = out["action_type"].clone()
                mu       = out["action_parms"]["mu"]
                sigma    = out["action_parms"]["sigma"]
                param_latent = out["param_latent"]

                ill = step.get("ill_action", [])
                if ill:
                    logits[0, torch.as_tensor(ill, device=device, dtype=torch.long)] = -1e9

                a_disc = int(step["action"])
                a_t    = torch.tensor([a_disc], device=device, dtype=torch.long)
                dist_disc  = Categorical(logits=logits)
                logp_disc  = dist_disc.log_prob(a_t)

                a_cont = step["act_param"]
                if self.continuous_action_mask[a_disc] == 0 or a_cont is None:
                    logp_cont = torch.zeros(1, device=device)
                else:
                    oh       = F.one_hot(a_t, num_classes=action_size).float()
                    has_mask = oh * self.continuous_action_mask.unsqueeze(0)
                    mu_full  = mu.detach().clone()
                    mu_full[0, a_disc] = float(a_cont)
                    logp_cont = (Normal(mu, sigma).log_prob(mu_full) * has_mask).sum(dim=-1)

                # refine head
                delta_mask_raw   = step.get("delta_mask", None)
                delta_sample_raw = step.get("delta_sample", None)
                if delta_mask_raw is not None and delta_sample_raw is not None:
                    dm = torch.tensor(delta_mask_raw,   device=device, dtype=torch.float32).unsqueeze(0)
                    ds = torch.tensor(delta_sample_raw, device=device, dtype=torch.float32).unsqueeze(0)
                    action_emb = self.action_embedding(a_t)
                    refine_in  = torch.cat([state_2d, action_emb, param_latent], dim=-1)
                    feat       = self.refine_backbone(refine_in)
                    mu_delta   = self.refine_mu_head(feat)
                    sigma_delta = torch.exp(self.refine_logsigma).unsqueeze(0).expand_as(mu_delta)
                    logp_delta  = (Normal(mu_delta, sigma_delta).log_prob(ds) * dm).sum(dim=-1)
                else:
                    logp_delta = torch.zeros(1, device=device)

                G_t = all_G_t[cursor]
                cursor += 1
                losses.append(-(logp_disc + logp_cont + logp_delta) * G_t)
                if entropy_coef != 0.0:
                    with torch.no_grad():
                        entropies.append(dist_disc.entropy().squeeze(0))

        policy_loss = torch.stack(losses).mean()
        if entropy_coef != 0.0 and entropies:
            policy_loss = policy_loss - entropy_coef * torch.stack(entropies).mean()

        self.optim.zero_grad()
        policy_loss.backward()
        if grad_clip and grad_clip > 0:
            nn.utils.clip_grad_norm_(
                list(self.HybridAction_policy_net.parameters()) +
                list(self.refine_backbone.parameters()) +
                list(self.refine_mu_head.parameters()) +
                [self.refine_logsigma],
                grad_clip,
            )
        self.optim.step()
        return float(policy_loss.detach().cpu())
