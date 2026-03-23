"""
Circuit environment for PSQASBench.
Adapted from CRLQAS/environment.py.

Key changes vs CRLQAS:
  - All imports are package-relative (no sys.path tricks).
  - Hamiltonian loaded via absolute path (conf['problem']['mol_data_dir'] + mol_file),
    not the hardcoded '../mol_data/' relative path.
  - SPSA / legacy bond-scan logic retained but mol_data path fixed everywhere.
"""

import copy
import os
from sys import stdout
from pathlib import Path

import numpy as np
import scipy.optimize
import torch
from qulacs import ParametricQuantumCircuit
from qulacs.gate import CNOT, RX, RY, RZ
try:
    from qulacs import QuantumStateGpu as QuantumState
except ImportError:
    from qulacs import QuantumState

from . import VQE as vc
from . import curricula
from .utils import dictionary_of_actions, to_tuple4


class CircuitEnv:

    def __init__(self, conf, device):
        self.num_qubits    = conf["env"]["num_qubits"]
        self.num_layers    = conf["env"]["num_layers"]
        self.random_halt   = int(conf["env"]["rand_halt"])
        self.n_shots       = conf["env"]["n_shots"]

        noise_models_all   = ["depolarizing", "two_depolarizing", "amplitude_damping"]
        noise_values_raw   = conf["env"]["noise_values"]
        if noise_values_raw != 0:
            idx = conf["env"]["noise_values"].index(",")
            self.noise_values = [float(noise_values_raw[1:idx]),
                                 float(noise_values_raw[idx + 1:-1])]
        else:
            self.noise_values = []
        self.noise_models  = noise_models_all[: len(self.noise_values)]
        self.phys_noise    = len(self.noise_models) > 0
        self.err_mitig     = conf["env"]["err_mitig"]

        self.ham_mapping   = conf["problem"]["mapping"]
        self.geometry      = conf["problem"]["geometry"].replace(" ", "_")
        self.mol           = conf["problem"]["ham_type"]

        self.fake_min_energy = conf["env"].get("fake_min_energy", None)
        self.fn_type         = conf["env"]["fn_type"]
        self.cnot_rwd_weight = conf["env"].get("cnot_rwd_weight", 1.0)

        self.noise_flag        = True
        self.state_with_angles = conf["agent"]["angles"]
        self.current_number_of_cnots = 0
        self.curriculum_dict   = {}

        # ── mol data: load via absolute path ──────────────────────────────────
        mol_data_dir = conf["problem"].get("mol_data_dir", "")
        mol_file     = conf["problem"]["mol_file"]
        self._mol_path = str(Path(mol_data_dir) / mol_file)

        __ham = np.load(self._mol_path, allow_pickle=True)
        self.hamiltonian  = __ham["hamiltonian"]
        self.weights      = __ham["weights"]
        eigvals           = __ham["eigvals"]
        self.energy_shift = float(__ham["energy_shift"])

        min_eig = (self.fake_min_energy if self.fake_min_energy is not None
                   else float(min(eigvals)) + self.energy_shift)
        self.min_eig  = min_eig
        self.min_energy = float(min(eigvals)) + self.energy_shift
        self.max_eig  = float(max(eigvals)) + self.energy_shift

        self.curriculum_dict[self.geometry[-3:]] = curricula.__dict__[
            conf["env"]["curriculum_type"]
        ](conf["env"], target_energy=min_eig)

        self.device = device
        self.ket    = QuantumState(self.num_qubits)
        self.done_threshold = conf["env"]["accept_err"]

        self.op_history    = []
        stdout.flush()
        self.state_size    = self.num_layers * self.num_qubits * (self.num_qubits + 3 + 3)
        self.step_counter  = -1
        self.prev_energy   = None
        self.moments       = [0] * self.num_qubits
        self.illegal_actions = [[]] * self.num_qubits
        self.energy        = 0
        self.action_size   = self.num_qubits * (self.num_qubits + 2)
        self.previous_action = [0, 0, 0, 0]

        if "non_local_opt" in conf:
            self.global_iters  = conf["non_local_opt"]["global_iters"]
            self.optim_method  = conf["non_local_opt"]["method"]
            self.optim_alg     = conf["non_local_opt"]["optim_alg"]
            if "maxfev" in conf["non_local_opt"]:
                self.maxfev = {"maxfev": int(conf["non_local_opt"]["maxfev"])}
            if "maxfev1" in conf["non_local_opt"]:
                self.maxfevs = {
                    "maxfev1": int(conf["non_local_opt"]["maxfev1"]),
                    "maxfev2": int(conf["non_local_opt"]["maxfev2"]),
                    "maxfev3": int(conf["non_local_opt"]["maxfev3"]),
                }
            if "a" in conf["non_local_opt"]:
                self.options = {
                    "a": conf["non_local_opt"]["a"],
                    "alpha": conf["non_local_opt"]["alpha"],
                    "c": conf["non_local_opt"]["c"],
                    "gamma": conf["non_local_opt"]["gamma"],
                    "beta_1": conf["non_local_opt"]["beta_1"],
                    "beta_2": conf["non_local_opt"]["beta_2"],
                }
                if "lamda" in conf["non_local_opt"]:
                    self.options["lamda"] = conf["non_local_opt"]["lamda"]
        else:
            self.global_iters = 0
            self.optim_method = None

        self.start_energy = self.min_eig + self.done_threshold

    # ── step ──────────────────────────────────────────────────────────────────

    def step(self, action, train_flag=True):
        next_state   = self.state.clone()
        self.step_counter += 1

        ctrl      = action[0]
        targ      = (action[0] + action[1]) % self.num_qubits
        rot_qubit = action[2]
        rot_axis  = action[3]
        self.action = action

        if rot_qubit < self.num_qubits:
            gate_tensor = self.moments[rot_qubit]
        elif ctrl < self.num_qubits:
            gate_tensor = max(self.moments[ctrl], self.moments[targ])

        if ctrl < self.num_qubits:
            next_state[gate_tensor][targ][ctrl] = 1
            self.op_history.append({"type": "cnot", "layer": int(gate_tensor),
                                    "ctrl": int(ctrl), "targ": int(targ)})
            self.current_number_of_cnots += 1
        elif rot_qubit < self.num_qubits:
            next_state[gate_tensor][self.num_qubits + rot_axis - 1][rot_qubit] = 1
            self.op_history.append({"type": "rot", "layer": int(gate_tensor),
                                    "q": int(rot_qubit), "axis": int(rot_axis)})

        if rot_qubit < self.num_qubits:
            self.moments[rot_qubit] += 1
        elif ctrl < self.num_qubits:
            m = max(self.moments[ctrl], self.moments[targ])
            self.moments[ctrl] = m + 1
            self.moments[targ] = m + 1

        self.current_action = action
        self.illegal_action_new()

        if self.optim_method == "scipy_each_step":
            thetas, nfev, opt_ang = self.scipy_optim(self.optim_alg)
            for i in range(self.num_layers):
                for j in range(3):
                    next_state[i][self.num_qubits + 3 + j, :] = thetas[i][j, :]

        self.state  = next_state.clone()
        energy, energy_noiseless = self.get_energy()
        if not self.noise_flag:
            energy = energy_noiseless

        self.energy = energy
        if energy < self.curriculum.lowest_energy and train_flag:
            self.curriculum.lowest_energy = copy.copy(energy)

        self.error          = float(abs(self.min_eig - energy))
        self.error_noiseless = float(abs(self.min_eig - energy_noiseless))
        rwd                 = self.reward_fn(energy)
        self.prev_energy    = np.copy(energy)

        energy_done  = int(self.error < self.done_threshold)
        layers_done  = self.step_counter == (self.num_layers - 1)
        done         = int(energy_done or layers_done)
        done_reason  = 1 if energy_done else (-1 if layers_done else 0)

        self.previous_action = copy.deepcopy(action)

        if self.random_halt and self.step_counter == self.halting_step:
            done = 1

        if done:
            self.curriculum.update_threshold(energy_done=energy_done)
            self.done_threshold = self.curriculum.get_current_threshold()
            self.curriculum_dict[str(self.current_bond_distance)] = copy.deepcopy(self.curriculum)

        if self.state_with_angles:
            return (next_state.view(-1).to(self.device),
                    torch.tensor(rwd, dtype=torch.float32, device=self.device),
                    done, done_reason)
        else:
            return (next_state[:, :self.num_qubits + 3].reshape(-1).to(self.device),
                    torch.tensor(rwd, dtype=torch.float32, device=self.device),
                    done, done_reason)

    # ── reset ─────────────────────────────────────────────────────────────────

    def reset(self):
        state = torch.zeros((self.num_layers, self.num_qubits + 6, self.num_qubits))
        self.op_history = []
        self.state      = state

        if self.random_halt:
            self.halting_step = int(
                np.clip(np.random.negative_binomial(n=70, p=0.573, size=1), 25, 70)[0]
            )

        self.current_number_of_cnots = 0
        self.current_action          = [self.num_qubits] * 4
        self.illegal_actions         = [[]] * self.num_qubits

        self.make_circuit(state)
        self.step_counter = -1
        self.moments      = [0] * self.num_qubits

        self.current_bond_distance = self.geometry[-3:]
        self.curriculum     = copy.deepcopy(self.curriculum_dict[str(self.current_bond_distance)])
        self.done_threshold = copy.deepcopy(self.curriculum.get_current_threshold())
        self.geometry       = self.geometry[:-3] + str(self.current_bond_distance)

        # reload mol data (same file — path is fixed at init time)
        __ham = np.load(self._mol_path, allow_pickle=True)
        self.hamiltonian  = __ham["hamiltonian"]
        self.weights      = __ham["weights"]
        eigvals           = __ham["eigvals"]
        self.energy_shift = float(__ham["energy_shift"])
        self.min_eig      = (self.fake_min_energy if self.fake_min_energy is not None
                             else float(min(eigvals)) + self.energy_shift)
        self.max_eig      = float(max(eigvals)) + self.energy_shift
        self.prev_energy  = self.get_energy(state)[1]

        if self.state_with_angles:
            return state.reshape(-1).to(self.device)
        else:
            return state[:, :self.num_qubits + 3].reshape(-1).to(self.device)

    # ── circuit construction ──────────────────────────────────────────────────

    def make_circuit(self, thetas=None):
        state  = self.state.clone()
        thetas = thetas if thetas is not None else state[:, self.num_qubits + 3:]
        circuit = ParametricQuantumCircuit(self.num_qubits)
        for i in range(self.num_layers):
            cnot_pos = np.where(state[i][: self.num_qubits] == 1)
            targ, ctrl = cnot_pos[0], cnot_pos[1]
            for r in range(len(ctrl)):
                circuit.add_gate(CNOT(ctrl[r], targ[r]))

            rot_pos = np.where(state[i][self.num_qubits: self.num_qubits + 3] == 1)
            rot_dirs, rot_qubits = rot_pos[0], rot_pos[1]
            for pos, r in enumerate(rot_dirs):
                q = rot_qubits[pos]
                if r == 0:
                    circuit.add_parametric_RX_gate(q, thetas[i][0][q])
                elif r == 1:
                    circuit.add_parametric_RY_gate(q, thetas[i][1][q])
                elif r == 2:
                    circuit.add_parametric_RZ_gate(q, thetas[i][2][q])
        return circuit

    # ── energy evaluation ─────────────────────────────────────────────────────

    def get_energy(self, thetas=None):
        circ         = self.make_circuit(thetas)
        qulacs_inst  = vc.Parametric_Circuit(self.num_qubits, self.noise_models, self.noise_values)
        noisy_circ   = qulacs_inst.construct_ansatz(self.state)
        expval_noisy     = vc._get_exp_val(self.num_qubits, noisy_circ, self.hamiltonian, self.phys_noise, self.err_mitig)
        expval_noiseless = vc._get_exp_val(self.num_qubits, circ, self.hamiltonian)
        shot_noise       = vc._get_shot_noise(self.weights, self.n_shots)
        energy           = expval_noisy + shot_noise + self.energy_shift
        energy_noiseless = expval_noiseless + self.energy_shift
        return energy, energy_noiseless

    # ── scipy optimiser ───────────────────────────────────────────────────────

    def scipy_optim(self, method, which_angles=[]):
        state    = self.state.clone()
        thetas   = state[:, self.num_qubits + 3:]
        rot_pos  = (state[:, self.num_qubits: self.num_qubits + 3] == 1).nonzero(as_tuple=True)
        angles   = thetas[rot_pos]

        qulacs_inst    = vc.Parametric_Circuit(self.num_qubits, self.noise_models, self.noise_values)
        qulacs_circuit = qulacs_inst.construct_ansatz(state)
        x0 = np.asarray(angles.cpu().detach())

        def cost(x):
            return vc.get_energy_qulacs(
                x, observable=self.hamiltonian, weights=self.weights,
                circuit=qulacs_circuit, n_qubits=self.num_qubits,
                energy_shift=self.energy_shift, n_shots=int(self.n_shots),
                phys_noise=self.phys_noise,
            )

        result = scipy.optimize.minimize(
            cost, x0=x0, method=method,
            options={"maxiter": self.global_iters},
        )
        thetas = state[:, self.num_qubits + 3:]
        thetas[rot_pos] = torch.tensor(result["x"], dtype=torch.float)
        return thetas, result["nfev"], result["x"]

    # ── reward function ───────────────────────────────────────────────────────

    def reward_fn(self, energy):
        fn = self.fn_type
        max_depth = self.step_counter == (self.num_layers - 1)

        if fn == "incremental_with_fixed_ends":
            if self.error < self.done_threshold:
                return 5.0
            elif max_depth:
                return -5.0
            else:
                return float(np.clip(
                    (self.prev_energy - energy) / abs(self.prev_energy - self.min_eig),
                    -1, 1,
                ))
        elif fn == "incremental_with_fixed_start":
            if self.error < self.done_threshold:
                return 5.0
            elif max_depth:
                return -5.0
            else:
                return float(np.clip(
                    (self.prev_energy - energy) / abs(self.start_energy - self.min_eig),
                    -1, 1,
                ))
        elif fn == "nive_fives":
            if self.error < self.done_threshold:
                return 5.0
            elif max_depth:
                return -5.0
            return 0.0
        elif fn == "incremental":
            return (self.prev_energy - energy) / abs(self.prev_energy - self.min_eig)
        elif fn == "incremental_clipped":
            return float(np.clip(
                (self.prev_energy - energy) / abs(self.prev_energy - self.min_eig),
                -1, 1,
            ))
        elif fn == "naive":
            return 1.0 * (self.error < self.done_threshold)
        elif fn == "staircase":
            return (0.2 * (self.error < 15 * self.done_threshold) +
                    0.4 * (self.error < 10 * self.done_threshold) +
                    0.6 * (self.error < 5 * self.done_threshold) +
                    1.0 * (self.error < self.done_threshold)) / 2.2
        elif fn == "end_energy":
            if self.error < self.done_threshold or max_depth:
                return (self.max_eig - energy) / (abs(self.min_eig) + abs(self.max_eig))
            return 0.0
        elif fn == "cnot_reduce":
            if self.error < self.done_threshold:
                return self.num_layers - self.cnot_rwd_weight * self.current_number_of_cnots
            elif max_depth:
                return -5.0
            else:
                return float(np.clip(
                    (self.prev_energy - energy) / abs(self.prev_energy - self.min_eig),
                    -1, 1,
                ))
        else:
            raise ValueError(f"Unknown reward fn: {fn}")

    # ── illegal action tracking ───────────────────────────────────────────────

    def illegal_action_new(self):
        action       = self.current_action
        ill          = self.illegal_actions
        ctrl         = action[0]
        targ         = (action[0] + action[1]) % self.num_qubits
        rot_qubit    = action[2]
        rot_axis     = action[3]

        def _first_empty(lst):
            for i in range(1, len(lst)):
                if len(lst[i]) == 0:
                    return i
            return None

        if ctrl < self.num_qubits:
            if sum(sum(l) for l in ill) != 0:
                for idx, ill_ac in enumerate(ill):
                    if not ill_ac:
                        continue
                    ill_targ = (ill_ac[0] + ill_ac[1]) % self.num_qubits
                    if ill_ac[2] == self.num_qubits:  # prev was CNOT
                        if ctrl in (ill_ac[0], ill_targ) or targ in (ill_ac[0], ill_targ):
                            ill[idx] = []
                    else:                              # prev was rotation
                        if ctrl == ill_ac[2] or targ == ill_ac[2]:
                            ill[idx] = []
                    e = _first_empty(ill)
                    if e is not None:
                        ill[e] = action
                    break
            else:
                ill[0] = action

        if rot_qubit < self.num_qubits:
            if sum(sum(l) for l in ill) != 0:
                for idx, ill_ac in enumerate(ill):
                    if not ill_ac:
                        continue
                    ill_targ = (ill_ac[0] + ill_ac[1]) % self.num_qubits
                    if ill_ac[0] == self.num_qubits:  # prev was rotation
                        if rot_qubit == ill_ac[2] and rot_axis != ill_ac[3]:
                            ill[idx] = []
                        elif rot_qubit != ill_ac[2]:
                            pass
                    else:                              # prev was CNOT
                        if rot_qubit in (ill_ac[0], ill_targ):
                            ill[idx] = []
                    e = _first_empty(ill)
                    if e is not None:
                        ill[e] = action
                    break
            else:
                ill[0] = action

        # de-duplicate and compact
        seen, write = set(), 0
        for i in range(len(ill)):
            if not ill[i]:
                continue
            k = to_tuple4(ill[i])
            if k in seen:
                ill[i] = []
            else:
                seen.add(k)
        for r in range(len(ill)):
            if ill[r]:
                if write != r:
                    ill[write] = list(ill[r])
                    ill[r] = []
                write += 1
        for k in range(write, len(ill)):
            ill[k] = []

        # decode to action indices
        action_dict    = dictionary_of_actions(self.num_qubits)
        illegal_decoded = [k for k, v in action_dict.items() if v in ill]
        self.illegal_actions = ill
        return illegal_decoded
