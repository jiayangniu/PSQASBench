"""
VQE energy evaluation utilities for PSQASBench.
Adapted from CRLQAS/VQE.py — noiseless path kept; noisy paths preserved but
not exercised in the current benchmark (noise_models = 0).
"""

import numpy as np
from typing import List, Callable, Optional
from scipy.optimize import OptimizeResult

from qulacs import ParametricQuantumCircuit, QuantumState, DensityMatrix
from qulacs.gate import CNOT, RX, RY, RZ
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
                      energy_shift, n_shots, phys_noise=False, which_angles=[]):
    param_count = circuit.get_parameter_count()
    if not list(which_angles):
        which_angles = np.arange(param_count)
    for i, j in enumerate(which_angles):
        circuit.set_parameter(j, angles[i])

    expval     = _get_exp_val(n_qubits, circuit, observable, phys_noise)
    shot_noise = _get_shot_noise(weights, n_shots)
    return expval + shot_noise + energy_shift


def _get_exp_val(n_qubits, circuit, op, phys_noise=False, err_mitig=0):
    if not phys_noise:
        state = QuantumState(n_qubits)
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
