"""
circuit.py — QuantumDARTS Macro Search circuit.

Implements the differentiable circuit from Section 3.2 of:
  Wu et al., "QuantumDARTS: Differentiable Quantum Architecture Search
  for Variational Quantum Algorithms", ICML 2023.

Key design:
  • Gate set G = {Rx(θ), Ry(θ), Rz(θ), I, CNOT}   (K = 5)
  • Each position (layer j, qubit i) selects a gate via Gumbel-Softmax
  • Architecture weights decomposed as α_ij = P_ij @ Q_ij  (Appendix H)
  • Rotation parameter θ_ij is shared across Rx/Ry/Rz at each position
  • CNOT convention: target = qubit i, control = (i+1) % n_qubits
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .gates import rx_2x2, ry_2x2, rz_2x2, apply_1q, apply_cnot, zero_state

GATE_NAMES = ['Rx', 'Ry', 'Rz', 'I', 'CNOT']
K = len(GATE_NAMES)   # 5


class QuantumDARTSCircuit(nn.Module):
    """
    n-qubit, m-layer differentiable VQE circuit.

    Parameters
    ----------
    n_qubits  : number of qubits
    n_layers  : number of circuit layers
    K_prime   : rank of architecture weight decomposition P@Q (paper default 3)
    """

    def __init__(self, n_qubits: int, n_layers: int, K_prime: int = 3):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.K_prime = K_prime

        # ── Architecture weights: α_ij = (P_ij @ Q_ij).squeeze(0)  ──────────
        # P_ij ∈ R^{1 × K'},  Q_ij ∈ R^{K' × K}
        self.P = nn.Parameter(
            torch.randn(n_layers, n_qubits, 1, K_prime, dtype=torch.float64) * 0.1
        )
        self.Q = nn.Parameter(
            torch.randn(n_layers, n_qubits, K_prime, K, dtype=torch.float64) * 0.1
        )

        # ── Rotation parameters θ_ij ∈ [-π, π]  ─────────────────────────────
        self.theta = nn.Parameter(
            (torch.rand(n_layers, n_qubits, dtype=torch.float64) * 2 - 1) * np.pi
        )

        # ── Precompute CNOT partner indices  ─────────────────────────────────
        # For qubit i: control = (i+1) % n, target = i
        self._cnot_controls = [(i + 1) % n_qubits for i in range(n_qubits)]

    # ─────────────────────────────────────────────────────────────────────────

    def arch_weights(self) -> torch.Tensor:
        """Return α ∈ R^{n_layers × n_qubits × K}."""
        return (self.P @ self.Q).squeeze(-2)   # (L, N, K)

    def sample_h(self, tau: float) -> torch.Tensor:
        """
        Sample Gumbel-Softmax gate selection weights.

        Forward: approximately one-hot as tau→0.
        Backward: smooth softmax gradient flows to P, Q.

        Returns: h ∈ [0,1]^{n_layers × n_qubits × K}, row sums = 1.
        """
        return F.gumbel_softmax(self.arch_weights(), tau=tau,
                                hard=False, dim=-1)

    def forward_sv(self, h: torch.Tensor) -> torch.Tensor:
        """
        Propagate |0…0⟩ through the circuit with gate weights h.

        Each position applies:
          |ψ⟩ ← Σ_k h[j,i,k] · G_k(θ_ij) |ψ⟩

        This weighted sum over gates preserves differentiability w.r.t.
        both h (→ architecture weights) and θ (→ rotation parameters).

        Args:
            h: (n_layers, n_qubits, K) gate selection weights (from sample_h)

        Returns:
            psi: (2^n,) complex128 output state vector
        """
        device = h.device
        psi = zero_state(self.n_qubits, device=device)

        # Cast h to complex for scalar multiplication with complex psi
        h_c = h.to(torch.complex128)

        for j in range(self.n_layers):
            for i in range(self.n_qubits):
                th = self.theta[j, i]
                ctrl = self._cnot_controls[i]

                # Compute 2×2 rotation matrices
                rx = rx_2x2(th)
                ry = ry_2x2(th)
                rz = rz_2x2(th)

                n = self.n_qubits
                # Weighted mixture of gate applications
                psi = (h_c[j, i, 0] * apply_1q(psi, rx, i, n)
                     + h_c[j, i, 1] * apply_1q(psi, ry, i, n)
                     + h_c[j, i, 2] * apply_1q(psi, rz, i, n)
                     + h_c[j, i, 3] * psi                          # Identity
                     + h_c[j, i, 4] * apply_cnot(psi, ctrl, i, n))

        return psi

    def energy(self, H: torch.Tensor, tau: float) -> torch.Tensor:
        """
        Compute VQE energy ⟨ψ|H|ψ⟩ with a fresh Gumbel sample.

        Args:
            H  : (2^n, 2^n) complex128 Hamiltonian tensor
            tau: Gumbel-Softmax temperature

        Returns:
            energy: real scalar tensor
        """
        h = self.sample_h(tau)
        psi = self.forward_sv(h)
        return (psi.conj() @ (H @ psi)).real

    def energy_fixed_h(self, h: torch.Tensor, H: torch.Tensor) -> torch.Tensor:
        """
        Compute VQE energy with pre-sampled h (used in inner θ loop).
        Gumbel samples are fixed; only θ changes.
        """
        psi = self.forward_sv(h)
        return (psi.conj() @ (H @ psi)).real

    # ─────────────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def discrete_circuit(self) -> list:
        """
        Extract the discrete circuit after training (argmax of α).

        Returns list of (gate_name, qubits, angle) tuples.
        Identity gates are dropped.
        """
        alpha = self.arch_weights()
        gate_idx = alpha.argmax(dim=-1)         # (L, N)
        theta = self.theta.cpu().numpy()
        gates = []
        for j in range(self.n_layers):
            for i in range(self.n_qubits):
                k = int(gate_idx[j, i])
                name = GATE_NAMES[k]
                if name == 'I':
                    continue
                elif name in ('Rx', 'Ry', 'Rz'):
                    gates.append((name, (i,), float(theta[j, i])))
                else:   # CNOT
                    ctrl = self._cnot_controls[i]
                    gates.append(('CNOT', (ctrl, i), None))
        return gates

    @torch.no_grad()
    def n_cnot(self) -> int:
        """Count CNOT gates in the discrete circuit."""
        return sum(1 for g, _, _ in self.discrete_circuit() if g == 'CNOT')
