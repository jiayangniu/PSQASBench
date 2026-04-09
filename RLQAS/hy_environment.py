"""
HyCircuitEnv — extends CircuitEnv for HyRLQAS-style hybrid action space.

Extra capabilities vs CircuitEnv:
  - step() accepts act_param (float|None) and refine_delta (Tensor|None)
  - act_param initialises the rotation angle before COBYLA (policy warm-start)
  - refine_delta applies Δθ increments to all prior rotation gates
  - delta_mask tracks which timesteps are rotation gates (used by RENEW head)
"""

import copy

import numpy as np
import torch

from .environment import CircuitEnv
from .utils import map_theta


class HyCircuitEnv(CircuitEnv):
    """CircuitEnv with hybrid action support (act_param + refine_delta)."""

    def __init__(self, conf, device, use_gpu_state: bool = True, shared_bvqe=None):
        super().__init__(conf, device, use_gpu_state=use_gpu_state, shared_bvqe=shared_bvqe)
        self.delta_mask = np.zeros(self.num_layers, dtype=np.float32)
        self.refine_mode = conf["env"].get("refine_mode", "delta")

    # ── reset ─────────────────────────────────────────────────────────────────

    def reset(self):
        self.delta_mask = np.zeros(self.num_layers, dtype=np.float32)
        return super().reset()

    # ── step ──────────────────────────────────────────────────────────────────

    def _apply_action_and_optimize(
        self,
        action,
        act_param=None,
        refine_delta=None,
        run_optimizer=True,
    ):
        """
        Apply one hybrid action and optionally run local optimizer.

        Args:
            action       : 4-tuple [ctrl, offset, rot_qubit, rot_axis]
            act_param    : float initial angle for rotation gate (None for CNOT)
            refine_delta : Tensor of shape [num_layers] — Δθ for prior rotations
            run_optimizer: if False, only updates structure/angles and defers
                           optimization + energy to external batched path
        """
        next_state = self.state.clone()
        self.step_counter += 1

        ctrl      = action[0]
        targ      = (action[0] + action[1]) % self.num_qubits
        rot_qubit = action[2]
        rot_axis  = action[3]
        self.action = action

        # ── illegal action check (before gate placement, matching HyRLQAS) ────
        self.current_action = action
        self.illegal_action_new()

        # ── determine layer via moments ────────────────────────────────────────
        if rot_qubit < self.num_qubits:
            gate_tensor = self.moments[rot_qubit]
            self.delta_mask[self.step_counter] = 1.0   # rotation at this step
        elif ctrl < self.num_qubits:
            gate_tensor = max(self.moments[ctrl], self.moments[targ])

        # ── write gate into state tensor ───────────────────────────────────────
        if ctrl < self.num_qubits:
            next_state[gate_tensor][targ][ctrl] = 1
            self.op_history.append({
                "type": "cnot", "layer": int(gate_tensor),
                "ctrl": int(ctrl), "targ": int(targ),
            })
            self.current_number_of_cnots += 1
        elif rot_qubit < self.num_qubits:
            # type marker (one-hot channel)
            next_state[gate_tensor][self.num_qubits + rot_axis - 1][rot_qubit] = 1
            # initial angle from policy (warm-start for COBYLA)
            theta = map_theta(act_param) if act_param is not None else 0.0
            next_state[gate_tensor][self.num_qubits + 3 + rot_axis - 1][rot_qubit] = theta
            self.op_history.append({
                "type": "rot", "layer": int(gate_tensor),
                "q": int(rot_qubit), "axis": int(rot_axis),
            })

        # ── update moments ─────────────────────────────────────────────────────
        if rot_qubit < self.num_qubits:
            self.moments[rot_qubit] += 1
        elif ctrl < self.num_qubits:
            m = max(self.moments[ctrl], self.moments[targ])
            self.moments[ctrl] = m + 1
            self.moments[targ] = m + 1

        # ── apply refine delta to prior rotation angles ────────────────────────
        if refine_delta is not None:
            if isinstance(refine_delta, torch.Tensor):
                delta_vec = refine_delta.detach().cpu().numpy().astype(np.float32)
            else:
                delta_vec = np.asarray(refine_delta, dtype=np.float32)
            upto = len(self.op_history) - 1          # exclude the gate just inserted
            for t in range(upto):
                rec = self.op_history[t]
                if rec["type"] != "rot":
                    continue
                L, q, a = rec["layer"], rec["q"], rec["axis"]
                angle_plane = self.num_qubits + 3 + a - 1
                old_theta = float(next_state[L][angle_plane][q].item())
                if self.refine_mode == "resample":
                    new_theta = float(delta_vec[t])
                else:
                    new_theta = old_theta + float(delta_vec[t])
                next_state[L][angle_plane][q] = map_theta(new_theta)

        if run_optimizer and self.optim_method == "scipy_each_step":
            # Update self.state first so the optimiser sees the newly added gate.
            self.state = next_state.clone()
            if self.optim_alg == "Rotosolve":
                thetas, _nfev, _opt = self.rotosolve_optim()
            elif str(self.optim_alg).upper() in {"SPSA", "ADAMSPSA"}:
                thetas, _nfev, _opt = self.spsa_optim(self.optim_alg)
            else:
                thetas, _nfev, _opt = self.scipy_optim(self.optim_alg)
            for i in range(self.num_layers):
                for j in range(3):
                    next_state[i][self.num_qubits + 3 + j, :] = thetas[i][j, :]

        self.state = next_state.clone()
        return next_state

    def step(self, action, act_param=None, refine_delta=None, train_flag=True):
        """
        Extended step for hybrid action space.
        """
        next_state = self._apply_action_and_optimize(
            action,
            act_param=act_param,
            refine_delta=refine_delta,
            run_optimizer=True,
        )

        energy, energy_noiseless = self.get_energy()
        return self._finalize_step(
            action,
            next_state,
            energy=energy,
            energy_noiseless=energy_noiseless,
            train_flag=train_flag,
        )

    def step_deferred_energy(
        self,
        action,
        act_param=None,
        refine_delta=None,
        run_optimizer=True,
    ):
        """Apply action + optimizer but defer energy evaluation."""
        return self._apply_action_and_optimize(
            action,
            act_param=act_param,
            refine_delta=refine_delta,
            run_optimizer=run_optimizer,
        )

    def finalize_step_with_energy(self, action, next_state, energy, energy_noiseless, train_flag=True):
        """Finalize reward/done bookkeeping with externally computed energies."""
        return self._finalize_step(
            action,
            next_state,
            energy=energy,
            energy_noiseless=energy_noiseless,
            train_flag=train_flag,
        )
