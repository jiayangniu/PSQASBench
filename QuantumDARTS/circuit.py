"""
circuit.py — faithful QuantumDARTS macro-search circuit.

Implements the macro-search candidate space described in:
  Wu et al., "QuantumDARTS: Differentiable Quantum Architecture Search
  for Variational Quantum Algorithms", ICML 2023.

Per search slot (layer j, target qubit i), the candidate set is:
  - Rz-Ry-Rz block on target i
  - Identity
  - CNOT(control -> i) for every topology-legal control qubit

For benchmarking / export, the Rz-Ry-Rz block is later lowered to primitive
RZ / RY / RZ gates, but the search itself remains paper-faithful.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .gates import apply_1q, apply_cnot, ry_2x2, rz_2x2, zero_state


ROT_BLOCK_NAME = "RzRyRz"
IDENTITY_NAME = "I"
CNOT_NAME = "CNOT"
ROT_BLOCK_COST = 3.0
CNOT_COST = 1.0
IDENTITY_COST = 0.0


def _legal_controls(n_qubits: int, target: int, connectivity: str) -> list[int]:
    if connectivity == "linear":
        controls = []
        if target - 1 >= 0:
            controls.append(target - 1)
        if target + 1 < n_qubits:
            controls.append(target + 1)
        return controls

    return [ctrl for ctrl in range(n_qubits) if ctrl != target]


class QuantumDARTSCircuit(nn.Module):
    """
    n-qubit, m-slot differentiable VQE circuit with faithful macro candidates.

    Parameters
    ----------
    n_qubits:
        Number of qubits.
    n_layers:
        Number of search slots per qubit.
    K_prime:
        Rank of the architecture-weight decomposition P @ Q.
    connectivity:
        ``all`` or ``linear``; determines which CNOT controls are legal.
    """

    def __init__(
        self,
        n_qubits: int,
        n_layers: int,
        K_prime: int = 3,
        connectivity: str = "all",
    ):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.K_prime = K_prime
        self.connectivity = str(connectivity).strip().lower()
        if self.connectivity not in {"all", "linear"}:
            raise ValueError(f"Unsupported connectivity '{connectivity}'.")

        self._candidate_specs_by_qubit: list[tuple[tuple[str, tuple[int, ...]], ...]] = []
        self._control_candidates: list[tuple[int, ...]] = []
        max_choices = 0

        for target in range(n_qubits):
            controls = tuple(_legal_controls(n_qubits, target, self.connectivity))
            specs = [(ROT_BLOCK_NAME, (target,)), (IDENTITY_NAME, (target,))]
            specs.extend((CNOT_NAME, (ctrl, target)) for ctrl in controls)
            specs_tuple = tuple(specs)
            self._candidate_specs_by_qubit.append(specs_tuple)
            self._control_candidates.append(controls)
            max_choices = max(max_choices, len(specs_tuple))

        self.max_choices = max_choices

        candidate_mask = torch.zeros(n_qubits, max_choices, dtype=torch.bool)
        gate_costs = torch.zeros(n_qubits, max_choices, dtype=torch.float64)
        for target, specs in enumerate(self._candidate_specs_by_qubit):
            candidate_mask[target, : len(specs)] = True
            for idx, (name, _qubits) in enumerate(specs):
                if name == ROT_BLOCK_NAME:
                    gate_costs[target, idx] = ROT_BLOCK_COST
                elif name == CNOT_NAME:
                    gate_costs[target, idx] = CNOT_COST
                else:
                    gate_costs[target, idx] = IDENTITY_COST

        self.register_buffer("candidate_mask", candidate_mask, persistent=False)
        self.register_buffer("gate_costs", gate_costs, persistent=False)

        self.P = nn.Parameter(
            torch.randn(n_layers, n_qubits, 1, K_prime, dtype=torch.float64) * 0.1
        )
        self.Q = nn.Parameter(
            torch.randn(n_layers, n_qubits, K_prime, max_choices, dtype=torch.float64) * 0.1
        )

        # Three parameters per paper-faithful Rz-Ry-Rz block.
        self.theta = nn.Parameter(
            (torch.rand(n_layers, n_qubits, 3, dtype=torch.float64) * 2.0 - 1.0) * math.pi
        )

    # ── Candidate metadata ──────────────────────────────────────────────────

    def candidate_specs(self, target: int) -> tuple[tuple[str, tuple[int, ...]], ...]:
        return self._candidate_specs_by_qubit[target]

    def candidate_count(self, target: int) -> int:
        return len(self._candidate_specs_by_qubit[target])

    def choice_spec(self, target: int, choice_idx: int) -> tuple[str, tuple[int, ...]]:
        return self._candidate_specs_by_qubit[target][choice_idx]

    # ── Architecture weights / sampling ─────────────────────────────────────

    def raw_arch_weights(self) -> torch.Tensor:
        return (self.P @ self.Q).squeeze(-2)

    def arch_weights(self) -> torch.Tensor:
        logits = self.raw_arch_weights()
        invalid_fill = torch.full_like(logits, -1.0e9)
        mask = self.candidate_mask.unsqueeze(0).expand_as(logits)
        return torch.where(mask, logits, invalid_fill)

    def arch_probs(self) -> torch.Tensor:
        return torch.softmax(self.arch_weights(), dim=-1)

    def sample_h(self, tau: float) -> torch.Tensor:
        return F.gumbel_softmax(self.arch_weights(), tau=tau, hard=False, dim=-1)

    def expected_gate_cost(self) -> torch.Tensor:
        probs = self.arch_probs()
        return (probs * self.gate_costs.unsqueeze(0)).sum()

    # ── State propagation ────────────────────────────────────────────────────

    def _apply_rot_block(
        self,
        psi: torch.Tensor,
        target: int,
        angles: torch.Tensor,
    ) -> torch.Tensor:
        n_qubits = self.n_qubits
        psi = apply_1q(psi, rz_2x2(angles[0]), target, n_qubits)
        psi = apply_1q(psi, ry_2x2(angles[1]), target, n_qubits)
        psi = apply_1q(psi, rz_2x2(angles[2]), target, n_qubits)
        return psi

    def forward_sv(self, h: torch.Tensor, theta: torch.Tensor | None = None) -> torch.Tensor:
        device = h.device
        theta_local = self.theta if theta is None else theta
        psi = zero_state(self.n_qubits, device=device)
        h_complex = h.to(torch.complex128)

        for layer in range(self.n_layers):
            for target in range(self.n_qubits):
                slot_angles = theta_local[layer, target]
                psi_next = torch.zeros_like(psi)
                for choice_idx, (name, qubits) in enumerate(self.candidate_specs(target)):
                    weight = h_complex[layer, target, choice_idx]
                    if name == ROT_BLOCK_NAME:
                        branch = self._apply_rot_block(psi, target, slot_angles)
                    elif name == IDENTITY_NAME:
                        branch = psi
                    else:
                        ctrl, targ = qubits
                        branch = apply_cnot(psi, ctrl, targ, self.n_qubits)
                    psi_next = psi_next + weight * branch
                psi = psi_next

        return psi

    def energy(self, H: torch.Tensor, tau: float) -> torch.Tensor:
        h = self.sample_h(tau)
        psi = self.forward_sv(h)
        return (psi.conj() @ (H @ psi)).real

    def energy_fixed_h(
        self,
        h: torch.Tensor,
        H: torch.Tensor,
        theta: torch.Tensor | None = None,
    ) -> torch.Tensor:
        psi = self.forward_sv(h, theta=theta)
        return (psi.conj() @ (H @ psi)).real

    def hard_selection(self, gate_idx: torch.Tensor) -> torch.Tensor:
        h_hard = torch.zeros(
            self.n_layers,
            self.n_qubits,
            self.max_choices,
            dtype=torch.float64,
            device=gate_idx.device,
        )
        for layer in range(self.n_layers):
            for target in range(self.n_qubits):
                h_hard[layer, target, int(gate_idx[layer, target])] = 1.0
        return h_hard

    # ── Discrete helpers ─────────────────────────────────────────────────────

    @torch.no_grad()
    def discrete_gate_idx(self) -> torch.Tensor:
        return self.arch_weights().argmax(dim=-1)

    @torch.no_grad()
    def discrete_macro_circuit(self) -> list[tuple[int, str, tuple[int, ...], tuple[float, ...] | None]]:
        gate_idx = self.discrete_gate_idx()
        theta_np = self.theta.detach().cpu().numpy()
        gates: list[tuple[int, str, tuple[int, ...], tuple[float, ...] | None]] = []
        for layer in range(self.n_layers):
            for target in range(self.n_qubits):
                name, qubits = self.choice_spec(target, int(gate_idx[layer, target]))
                if name == IDENTITY_NAME:
                    continue
                if name == ROT_BLOCK_NAME:
                    gates.append(
                        (
                            layer,
                            name,
                            qubits,
                            tuple(float(x) for x in theta_np[layer, target]),
                        )
                    )
                else:
                    gates.append((layer, name, qubits, None))
        return gates

    @torch.no_grad()
    def n_cnot(self) -> int:
        return sum(1 for _layer, name, _qubits, _payload in self.discrete_macro_circuit() if name == CNOT_NAME)
