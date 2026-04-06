"""
VQE energy evaluation utilities for PSQASBench.
Adapted from CRLQAS/VQE.py — noiseless path kept; noisy paths preserved but
not exercised in the current benchmark (noise_models = 0).
"""

import numpy as np
import torch
from typing import List, Callable, Optional
from scipy.optimize import OptimizeResult

from qulacs import ParametricQuantumCircuit, DensityMatrix, QuantumState as QuantumStateCPU
from qulacs.gate import CNOT, RX, RY, RZ
try:
    from qulacs import QuantumStateGpu as QuantumStateGPU
except ImportError:
    QuantumStateGPU = QuantumStateCPU
# Default alias used by single-env (sequential) code paths.
QuantumState = QuantumStateGPU
from qulacs.gate import (DepolarizingNoise, TwoQubitDepolarizingNoise,
                          AmplitudeDampingNoise)


# ── Circuit construction ──────────────────────────────────────────────────────

class Parametric_Circuit:
    def __init__(self, n_qubits, noise_models=None, noise_values=None):
        self.n_qubits     = n_qubits
        self.noise_models = noise_models or []
        self.noise_values = noise_values or []
        self.ansatz       = ParametricQuantumCircuit(n_qubits)

    def construct_ansatz(self, state):
        if len(self.noise_models) >= 1:
            channels_1 = _get_noise_channels(self.noise_models[0],
                                             self.n_qubits, self.noise_values[0])
        if len(self.noise_models) == 3:
            channels_3 = _get_noise_channels(self.noise_models[2],
                                             self.n_qubits, self.noise_values[2])

        for local_state in state:
            thetas  = local_state[self.n_qubits + 3:]
            rot_pos  = (local_state[self.n_qubits: self.n_qubits + 3] == 1).nonzero(as_tuple=True)
            cnot_pos = (local_state[:self.n_qubits] == 1).nonzero(as_tuple=True)

            targ, ctrl = cnot_pos[0], cnot_pos[1]
            if len(ctrl):
                for r in range(len(ctrl)):
                    self.ansatz.add_gate(CNOT(ctrl[r], targ[r]))
                    if len(self.noise_models) >= 2:
                        self.ansatz.add_gate(
                            TwoQubitDepolarizingNoise(ctrl[r], targ[r], self.noise_values[1])
                        )

            rot_dirs, rot_qubits = rot_pos[0], rot_pos[1]
            if len(rot_qubits):
                for pos, r in enumerate(rot_dirs):
                    q = rot_qubits[pos]
                    if r == 0:
                        self.ansatz.add_parametric_RX_gate(q, thetas[0][q])
                    elif r == 1:
                        self.ansatz.add_parametric_RY_gate(q, thetas[1][q])
                    elif r == 2:
                        self.ansatz.add_parametric_RZ_gate(q, thetas[2][q])

                    if 1 <= len(self.noise_values) < 3:
                        self.ansatz.add_gate(channels_1[q])
                    elif len(self.noise_values) >= 3:
                        self.ansatz.add_gate(channels_1[q])
                        self.ansatz.add_gate(channels_3[q])

        return self.ansatz


# ── Energy evaluation ─────────────────────────────────────────────────────────

def get_energy_qulacs(angles, observable, weights, circuit, n_qubits,
                      energy_shift, n_shots, phys_noise=False, which_angles=[],
                      state=None):
    param_count = circuit.get_parameter_count()
    if not list(which_angles):
        which_angles = np.arange(param_count)
    for i, j in enumerate(which_angles):
        circuit.set_parameter(j, angles[i])

    expval     = _get_exp_val(n_qubits, circuit, observable, phys_noise, state=state)
    shot_noise = _get_shot_noise(weights, n_shots)
    return expval + shot_noise + energy_shift


def _get_exp_val(n_qubits, circuit, op, phys_noise=False, err_mitig=0, state=None):
    if not phys_noise:
        if state is None:
            state = QuantumState(n_qubits)
        else:
            state.set_zero_state()
        circuit.update_quantum_state(state)
        psi = state.get_vector()
        return (np.conj(psi).T @ op @ psi).real
    else:
        dm = DensityMatrix(n_qubits)
        circuit.update_quantum_state(dm)
        rho = dm.get_matrix()
        if err_mitig == 0:
            return np.real(np.trace(op @ rho))
        else:
            return np.real(np.trace(op @ rho @ rho) / np.trace(rho @ rho))


def _get_shot_noise(weights, n_shots):
    if n_shots <= 0:
        return 0.0
    w1 = weights[np.abs(weights) > 0.05]
    w2 = weights[np.abs(weights) <= 0.05]
    noise  = w1.real @ np.random.normal(0, (10 * n_shots) ** -0.5, len(w1))
    noise += w2.real @ np.random.normal(0, n_shots ** -0.5, len(w2))
    return noise


def _get_noise_channels(model_name, n_qubits, error_prob):
    _map = {
        "depolarizing":    DepolarizingNoise,
        "amplitude_damping": AmplitudeDampingNoise,
        "two_depolarizing": TwoQubitDepolarizingNoise,
    }
    model = _map[model_name]
    return [model(q, error_prob) for q in range(n_qubits)]


# ── Batched GPU VQE ───────────────────────────────────────────────────────────

class BatchedVQE:
    """Evaluate B circuits simultaneously on GPU via PyTorch tensor contractions.

    Replaces B sequential Qulacs calls with one batched matmul kernel.
    Gates are applied as einsum contractions on the (B, 2^n) state tensor;
    CNOT uses index-swap arithmetic — no unitary matrix is ever materialised.

    Circuit structure is read from CircuitEnv.state (same layer-then-type order
    as make_circuit / construct_ansatz), so results are numerically identical to
    the Qulacs noiseless path.
    """

    def __init__(self, n_qubits: int, hamiltonian, energy_shift: float, device):
        self.n      = n_qubits
        self.dim    = 1 << n_qubits
        self.device = device
        self.set_problem(hamiltonian, energy_shift)
        self._cnot_cache: dict = {}

    def set_problem(self, hamiltonian, energy_shift: float):
        """Update Hamiltonian / energy shift used for batched energy evaluation."""
        self.shift = float(energy_shift)
        self.H = torch.tensor(
            np.array(hamiltonian, dtype=np.complex64),
            device=self.device,
        )

    # ── public API ─────────────────────────────────────────────────────────────

    def eval_batch(self, state, angle_batch):
        """Evaluate B circuits sharing the same gate structure but different angles.

        Parameters
        ----------
        state       : (L, N+6, N) CircuitEnv state tensor (any device)
        angle_batch : (B, n_rot_params) float32 tensor on self.device.
                      Column j → j-th rotation gate in layer-major, row-major order
                      (same ordering as scipy_optim / rotosolve_optim rot_pos).

        Returns
        -------
        (B,) float32 energy tensor on self.device
        """
        n  = self.n
        B  = angle_batch.shape[0]
        st = state.to(self.device)

        psi = torch.zeros(B, self.dim, dtype=torch.complex64, device=self.device)
        psi[:, 0] = 1.0          # |0...0⟩

        rot_idx = 0
        for l in range(st.shape[0]):
            layer = st[l]
            # CNOTs
            cnot_targs, cnot_ctrls = (layer[:n] == 1).nonzero(as_tuple=True)
            for targ, ctrl in zip(cnot_targs.tolist(), cnot_ctrls.tolist()):
                psi = self._cnot(psi, ctrl, targ)
            # Rotations
            rot_axes, rot_qubits = (layer[n:n+3] == 1).nonzero(as_tuple=True)
            for axis_0, q in zip(rot_axes.tolist(), rot_qubits.tolist()):
                psi = self._rot(psi, axis_0 + 1, q, angle_batch[:, rot_idx])
                rot_idx += 1

        # E_b = ⟨ψ_b|H|ψ_b⟩ via einsum (no intermediate (B,dim,dim) tensor)
        energies = torch.real(torch.einsum('bi,ij,bj->b', psi.conj(), self.H, psi))
        return energies + self.shift

    # ── gate helpers ──────────────────────────────────────────────────────────

    def _rot(self, psi, axis: int, q: int, thetas):
        """Apply R_{axis}(θ_b) on qubit q.  psi: (B,2^n), thetas: (B,) float32."""
        B   = psi.shape[0]
        ph  = thetas.to(torch.float32) * 0.5
        c   = torch.cos(ph).to(torch.complex64)   # (B,)
        s   = torch.sin(ph).to(torch.complex64)   # (B,)

        if axis == 1:        # RX = [[c, -is], [-is, c]]
            G = torch.stack([torch.stack([c, -1j*s], 1),
                             torch.stack([-1j*s,  c], 1)], 1)          # (B,2,2)
        elif axis == 2:      # RY = [[c, -s], [s, c]]  (real entries)
            G = torch.stack([torch.stack([c, -s], 1),
                             torch.stack([s,  c], 1)], 1)
        else:                # RZ = diag(e^{-iθ/2}, e^{+iθ/2})
            z = torch.zeros(B, dtype=torch.complex64, device=self.device)
            G = torch.stack([torch.stack([c - 1j*s, z       ], 1),
                             torch.stack([z,        c + 1j*s], 1)], 1)

        # Reshape psi to (B, 2^{n-q-1}, 2, 2^q): upper | qubit q | lower
        psi_r   = psi.reshape(B, 1 << (self.n - q - 1), 2, 1 << q)
        psi_out = torch.einsum('bji,buil->bujl', G, psi_r)
        return psi_out.reshape(B, self.dim)

    def _cnot(self, psi, ctrl: int, targ: int):
        """CNOT(ctrl, targ) via index-swap — same for all B, no matrix formed."""
        key = (ctrl, targ)
        if key not in self._cnot_cache:
            idx    = torch.arange(self.dim, device=self.device)
            src    = idx[((idx >> ctrl) & 1 == 1) & ((idx >> targ) & 1 == 0)]
            self._cnot_cache[key] = (src, src ^ (1 << targ))
        src, dst   = self._cnot_cache[key]
        psi_new    = psi.clone()
        psi_new[:, src] = psi[:, dst]
        psi_new[:, dst] = psi[:, src]
        return psi_new


# ── SPSA optimisers (kept for completeness, used when optim_alg != COBYLA) ───

def _spsa_lr(epoch, a, A, alpha):
    return a / (epoch + 1.0 + A) ** alpha

def _spsa_ck(epoch, c, gamma):
    return c / (epoch + 1.0) ** gamma

def _spsa_grad(fun, params, n, ck):
    delta = np.random.choice([-1, 1], size=n)
    return (fun(params + ck * delta) - fun(params - ck * delta)) / (2 * ck * delta)

def _adam_step(epoch, grad, m, v, b1, b2, eps=1e-8):
    m = b1 * m + (1 - b1) * grad
    v = b2 * v + (1 - b2) * grad ** 2
    return m / (1 - b1 ** (epoch + 1)) / (np.sqrt(v / (1 - b2 ** (epoch + 1))) + eps), m, v
