"""
model.py — GPT-style model for offline GQE training.

This module vendors the minimal model logic needed from the external
Q-GESolver repository so PSQASBench can run GQE without depending on a
separate sibling checkout at runtime.

Sampling follows the GPT-QE sign convention:

    probs = softmax(-logits / temperature)

so that more negative cumulative logits correspond to lower predicted energy.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    vocab_size: int
    block_size: int
    n_layer: int
    n_head: int
    n_embd: int
    dropout: float = 0.1
    bias: bool = False


class TransformerBlock(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=config.n_embd,
            num_heads=config.n_head,
            dropout=config.dropout,
            batch_first=True,
            bias=config.bias,
        )
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias),
            nn.Dropout(config.dropout),
        )
        self.ln_2 = nn.LayerNorm(config.n_embd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        t = x.size(1)
        causal_mask = torch.triu(
            torch.ones(t, t, dtype=torch.bool, device=x.device),
            diagonal=1,
        )
        attn_out, _ = self.attn(x, x, x, attn_mask=causal_mask, need_weights=False)
        x = self.ln_1(x + attn_out)
        x = self.ln_2(x + self.mlp(x))
        return x


class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(
            {
                "wte": nn.Embedding(config.vocab_size, config.n_embd),
                "wpe": nn.Embedding(config.block_size, config.n_embd),
                "drop": nn.Dropout(config.dropout),
                "h": nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layer)]),
                "ln_f": nn.LayerNorm(config.n_embd),
            }
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=config.bias)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        bsz, seq_len = idx.size()
        pos = torch.arange(seq_len, dtype=torch.long, device=idx.device).unsqueeze(0).expand(bsz, seq_len)
        tok_emb = self.transformer["wte"](idx)
        pos_emb = self.transformer["wpe"](pos)
        x = self.transformer["drop"](tok_emb + pos_emb)
        for block in self.transformer["h"]:
            x = block(x)
        x = self.transformer["ln_f"](x)
        return self.lm_head(x)

    def configure_optimizers(
        self,
        *,
        weight_decay: float,
        learning_rate: float,
        betas: tuple[float, float],
    ) -> torch.optim.Optimizer:
        decay_params = []
        no_decay_params = []
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if param.dim() >= 2 and "ln" not in name and "bias" not in name:
                decay_params.append(param)
            else:
                no_decay_params.append(param)
        return torch.optim.AdamW(
            [
                {"params": decay_params, "weight_decay": weight_decay},
                {"params": no_decay_params, "weight_decay": 0.0},
            ],
            lr=learning_rate,
            betas=betas,
        )


class GPTQE(GPT):
    """
    GPT-QE objective used by the current PSQASBench GQE integration.

    The model is trained so that the cumulative sum of the logits associated
    with the actually chosen tokens matches the true prefix energies.
    """

    def calculate_loss(self, tokens: torch.Tensor, energies: torch.Tensor) -> torch.Tensor:
        current_tokens = tokens[:, :-1]
        next_tokens = tokens[:, 1:]
        logits = self(current_tokens)
        next_token_mask = F.one_hot(next_tokens.long(), num_classes=self.config.vocab_size).float()
        next_token_logits = (logits * next_token_mask).sum(dim=2)
        cumsum_logits = torch.cumsum(next_token_logits, dim=1)
        return torch.mean(torch.square(cumsum_logits - energies))

    @torch.no_grad()
    def generate(
        self,
        *,
        n_sequences: int,
        max_new_tokens: int,
        temperature: float = 1.0,
        device: str | torch.device = "cpu",
    ) -> tuple[torch.Tensor, torch.Tensor]:
        idx = torch.zeros((n_sequences, 1), dtype=torch.long, device=device)
        total_logits = torch.zeros((n_sequences, 1), dtype=torch.float32, device=device)
        block_size = self.config.block_size

        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= block_size else idx[:, -block_size:]
            logits = self(idx_cond)[:, -1, :]
            # Token 0 is BOS and should never be sampled.
            logits[:, 0] = float("inf")
            probs = F.softmax(-logits / max(float(temperature), 1e-6), dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            total_logits += torch.gather(logits, dim=1, index=idx_next)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx, total_logits
