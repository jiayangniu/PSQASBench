"""
gates.py — Efficient quantum gate application for QuantumDARTS.

Uses state-vector propagation (O(2^n) per gate) rather than full unitary
matrix products (O(4^n)) for efficiency.

Qubit ordering convention: qubit 0 = LSB (bit position 0 in the integer index).
This matches Qulacs and BatchedVQE so that the DARTS search energy evaluates
the same Hamiltonian convention as the fixed-circuit optimisation step.
"""

import torch
import numpy as np
from typing import Tuple


# ──────────────────────────────────────────────────────────────────────────────
# 2×2 gate matrices (differentiable w.r.t. theta)
# ──────────────────────────────────────────────────────────────────────────────

def rx_2x2(theta: torch.Tensor) -> torch.Tensor:
    """RX(θ) = exp(-iθ/2·X), shape (2,2) complex128."""
    c = torch.cos(theta / 2)
    s = torch.sin(theta / 2)
    z = torch.zeros_like(c)
    real = torch.stack([c, z, z, c]).reshape(2, 2)
    imag = torch.stack([z, -s, -s, z]).reshape(2, 2)
    return torch.complex(real, imag)


def ry_2x2(theta: torch.Tensor) -> torch.Tensor:
    """RY(θ) = exp(-iθ/2·Y), shape (2,2) complex128."""
    c = torch.cos(theta / 2)
    s = torch.sin(theta / 2)
    z = torch.zeros_like(c)
    real = torch.stack([c, -s, s, c]).reshape(2, 2)
    imag = torch.stack([z, z, z, z]).reshape(2, 2)
    return torch.complex(real, imag)


def rz_2x2(theta: torch.Tensor) -> torch.Tensor:
    """RZ(θ) = exp(-iθ/2·Z), shape (2,2) complex128."""
    c = torch.cos(theta / 2)
    s = torch.sin(theta / 2)
    z = torch.zeros_like(c)
    real = torch.stack([c, z, z, c]).reshape(2, 2)
    imag = torch.stack([-s, z, z, s]).reshape(2, 2)
    return torch.complex(real, imag)


# ──────────────────────────────────────────────────────────────────────────────
# State-vector gate application
# ──────────────────────────────────────────────────────────────────────────────

def apply_1q(psi: torch.Tensor, gate: torch.Tensor,
             qubit: int, n_qubits: int) -> torch.Tensor:
    """
    Apply a (2,2) complex gate to `qubit` of a 2^n state vector.

    LSB convention: qubit q occupies bit position q, so it maps to
    axis (n_qubits - 1 - q) in the rank-n reshaped tensor.

    Algorithm:
      1. Reshape psi → (2, 2, …, 2)  [n_qubits axes]
      2. Permute so qubit axis is first
      3. Apply gate as (2×2) @ (2, 2^{n-1}) matrix product
      4. Invert permutation, reshape to (2^n,)
    """
    psi_r = psi.reshape([2] * n_qubits)
    axis = n_qubits - 1 - qubit
    axes = [axis] + [a for a in range(n_qubits) if a != axis]
    psi_p = psi_r.permute(axes)
    d_rest = 2 ** (n_qubits - 1)
    psi_new = (gate @ psi_p.reshape(2, d_rest)).reshape(psi_p.shape)
    inv = [0] * n_qubits
    for new, old in enumerate(axes):
        inv[old] = new
    return psi_new.permute(inv).reshape(-1)


def apply_cnot(psi: torch.Tensor, control: int, target: int,
               n_qubits: int) -> torch.Tensor:
    """
    Apply CNOT (control, target) to state vector via index gather.
    O(2^n), differentiable via gather.

    LSB convention: qubit q = bit position q in the integer index.
    """
    dim = psi.shape[0]
    idx = torch.arange(dim, device=psi.device)
    ctrl_bit = (idx >> control) & 1
    flipped = torch.where(ctrl_bit.bool(), idx ^ (1 << target), idx)
    return psi[flipped]


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def zero_state(n_qubits: int, device=None) -> torch.Tensor:
    """Return |0…0⟩ as a complex128 state vector of length 2^n."""
    psi = torch.zeros(2 ** n_qubits, dtype=torch.complex128, device=device)
    psi[0] = 1.0
    return psi
