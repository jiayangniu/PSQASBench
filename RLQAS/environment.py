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

from . import VQE as vc
from . import curricula
from .utils import get_action_dict, to_tuple4


class CircuitEnv:

    def __init__(self, conf, device, use_gpu_state: bool = True, shared_bvqe=None):
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
        self._loaded_bond_distance = self.geometry[-3:]

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
        # use_gpu_state=False forces CPU QuantumState even when QuantumStateGpu
        # is available.  Parallel envs must use CPU to avoid CUDA-stream
        # contention (K threads sharing the default stream would serialize).
        _QS = vc.QuantumStateGPU if use_gpu_state else vc.QuantumStateCPU
        self.ket = _QS(self.num_qubits)

        # BatchedVQE always runs on GPU (if available) regardless of use_gpu_state.
        # The state vector and H are tiny for 4-8 qubits; GPU matmul batches all
        # Rotosolve probes in one kernel launch instead of N sequential Qulacs calls.
        if isinstance(self.device, torch.device):
            _runner_dev = self.device
        else:
            _runner_dev = torch.device(self.device)
        if _runner_dev.type == "cuda" and torch.cuda.is_available():
            _bvqe_dev = _runner_dev
        else:
            _bvqe_dev = torch.device("cpu")
        if shared_bvqe is not None:
            self._batched_vqe = shared_bvqe
            self._owns_batched_vqe = False
        else:
            self._batched_vqe = vc.BatchedVQE(
                self.num_qubits, self.hamiltonian, self.energy_shift, _bvqe_dev
            )
            self._owns_batched_vqe = True

        self.done_threshold = conf["env"]["accept_err"]

        self.op_history    = []
        stdout.flush()
        self.state_size    = self.num_layers * self.num_qubits * (self.num_qubits + 3 + 3)
        self.step_counter  = -1
        self.prev_energy   = None
        self.moments       = [0] * self.num_qubits
        self.illegal_actions = [[]] * self.num_qubits
        self.energy        = 0
        connectivity = conf["env"].get("connectivity", "all")
        self.action_dict = get_action_dict(self.num_qubits, connectivity)
        self.action_size = len(self.action_dict)
        self.previous_action = [0, 0, 0, 0]

        if "non_local_opt" in conf:
            # `global_iters` is required by COBYLA / SPSA / PSRAdam, but not by
            # Rotosolve. Default to 0 so Rotosolve configs can omit it cleanly.
            self.global_iters  = int(conf["non_local_opt"].get("global_iters", 0))
            self.optim_method  = conf["non_local_opt"]["method"]
            self.optim_alg     = conf["non_local_opt"]["optim_alg"]
            self.rotosolve_sweeps = int(conf["non_local_opt"].get("rotosolve_sweeps", 1))
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
                }
                for key in ("alpha", "c", "gamma", "beta_1", "beta_2", "lamda"):
                    if key in conf["non_local_opt"]:
                        self.options[key] = conf["non_local_opt"][key]
            elif "lr" in conf["non_local_opt"]:
                self.options = {
                    "lr": conf["non_local_opt"]["lr"],
                }
                for key in ("beta_1", "beta_2"):
                    if key in conf["non_local_opt"]:
                        self.options[key] = conf["non_local_opt"][key]
            # GPU-COBYLA: use BatchedVQE (batch_size=1) inside scipy.optimize
            # instead of Qulacs CPU.  No external batching across envs.
            self._batched_cobyla = bool(int(
                conf["non_local_opt"].get("global_batched_cobyla", 0)
            ))
            # CPU-optimizer: force Qulacs CPU for Rotosolve / PSRAdam energy eval.
            # Used for benchmarking GPU vs CPU cost.  Overrides GPU paths.
            self._cpu_optimizer = bool(int(
                conf["non_local_opt"].get("cpu_optimizer", 0)
            ))
        else:
            self.global_iters = 0
            self.optim_method = None
            self._batched_cobyla = False
            self._cpu_optimizer  = False

        self.start_energy = self.min_eig + self.done_threshold

    @staticmethod
    def _auto_spsa_batch_size(n_params: int) -> int:
        """Heuristic multi-direction batch size for SPSA-style optimisers.

        We bias toward larger multi-direction batches because the benchmark is
        optimized for GPU-rich setups where reducing wall-clock is more
        important than minimizing per-step probe count.

        Rule:
          - for small circuits (<= 8 active parameters), use all directions;
          - otherwise use roughly half the active parameters;
          - cap at 32 probe directions to keep memory and launch cost bounded.
        """
        n = max(1, int(n_params))
        if n <= 8:
            return n
        return max(1, min(32, int(np.ceil(n / 2.0))))

    # ── step ──────────────────────────────────────────────────────────────────

    def _apply_action_and_optimize(self, action, run_optimizer=True):
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

        if run_optimizer and self.optim_method == "scipy_each_step":
            # Update self.state first so the optimiser sees the newly added gate.
            self.state = next_state.clone()
            _alg = str(self.optim_alg).upper()
            if getattr(self, "_cpu_optimizer", False):
                # CPU (Qulacs) path — used for CPU vs GPU benchmarking
                if _alg == "ROTOSOLVE":
                    thetas, nfev, opt_ang = self.rotosolve_optim_cpu()
                elif _alg == "PSRADAM":
                    thetas, nfev, opt_ang = self.psr_adam_optim_cpu()
                else:
                    thetas, nfev, opt_ang = self.scipy_optim(self.optim_alg)
            elif _alg == "ROTOSOLVE":
                thetas, nfev, opt_ang = self.rotosolve_optim()
            elif _alg in {"SPSA", "ADAMSPSA"}:
                thetas, nfev, opt_ang = self.spsa_optim(self.optim_alg)
            elif _alg == "PSRADAM":
                thetas, nfev, opt_ang = self.psr_adam_optim()
            elif getattr(self, "_batched_cobyla", False):
                thetas, nfev, opt_ang = self.scipy_optim_gpu(self.optim_alg)
            else:
                thetas, nfev, opt_ang = self.scipy_optim(self.optim_alg)
            for i in range(self.num_layers):
                for j in range(3):
                    next_state[i][self.num_qubits + 3 + j, :] = thetas[i][j, :]

        self.state  = next_state.clone()
        return next_state

    def _finalize_step(self, action, next_state, energy, energy_noiseless, train_flag=True):
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

    def step(self, action, train_flag=True):
        next_state = self._apply_action_and_optimize(action, run_optimizer=True)
        energy, energy_noiseless = self.get_energy()
        return self._finalize_step(
            action, next_state, energy=energy, energy_noiseless=energy_noiseless, train_flag=train_flag
        )

    def step_deferred_energy(self, action, run_optimizer=True):
        """Apply action + optimiser and update self.state, but defer energy eval.

        Used by parallel runner to evaluate multiple env energies on GPU together.
        """
        return self._apply_action_and_optimize(action, run_optimizer=run_optimizer)

    def finalize_step_with_energy(self, action, next_state, energy, energy_noiseless, train_flag=True):
        """Finalize reward/done bookkeeping using externally computed energies."""
        return self._finalize_step(
            action, next_state, energy=energy, energy_noiseless=energy_noiseless, train_flag=train_flag
        )

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

        self.step_counter = -1
        self.moments      = [0] * self.num_qubits

        self.current_bond_distance = self.geometry[-3:]
        self.curriculum     = copy.deepcopy(self.curriculum_dict[str(self.current_bond_distance)])
        self.done_threshold = copy.deepcopy(self.curriculum.get_current_threshold())
        self.geometry       = self.geometry[:-3] + str(self.current_bond_distance)

        # Reload only when bond-distance key changed (single-molecule runs avoid
        # redundant disk I/O and GPU tensor re-init every episode).
        if self.current_bond_distance != self._loaded_bond_distance:
            __ham = np.load(self._mol_path, allow_pickle=True)
            self.hamiltonian  = __ham["hamiltonian"]
            self.weights      = __ham["weights"]
            eigvals           = __ham["eigvals"]
            self.energy_shift = float(__ham["energy_shift"])
            if self._owns_batched_vqe:
                self._batched_vqe.set_problem(self.hamiltonian, self.energy_shift)
            self.min_eig      = (self.fake_min_energy if self.fake_min_energy is not None
                                 else float(min(eigvals)) + self.energy_shift)
            self.min_energy   = float(min(eigvals)) + self.energy_shift
            self.max_eig      = float(max(eigvals)) + self.energy_shift
            self._loaded_bond_distance = self.current_bond_distance
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
        circ             = self.make_circuit(thetas)
        expval_noiseless = vc._get_exp_val(self.num_qubits, circ, self.hamiltonian,
                                           state=self.ket)
        energy_noiseless = expval_noiseless + self.energy_shift
        shot_noise       = vc._get_shot_noise(self.weights, self.n_shots)
        if self.phys_noise:
            qulacs_inst = vc.Parametric_Circuit(self.num_qubits, self.noise_models, self.noise_values)
            noisy_circ  = qulacs_inst.construct_ansatz(self.state)
            expval_noisy = vc._get_exp_val(self.num_qubits, noisy_circ, self.hamiltonian,
                                           phys_noise=True, err_mitig=self.err_mitig)
            energy       = expval_noisy + shot_noise + self.energy_shift
        else:
            energy = energy_noiseless + shot_noise
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
                state=self.ket,
            )

        result = scipy.optimize.minimize(
            cost, x0=x0, method=method,
            options={"maxiter": self.global_iters},
        )
        thetas = state[:, self.num_qubits + 3:]
        thetas[rot_pos] = torch.tensor(result["x"], dtype=torch.float)
        return thetas, result["nfev"], result["x"]

    def scipy_optim_gpu(self, method):
        """COBYLA (or any scipy optimizer) using BatchedVQE (GPU) for energy eval.

        Drop-in replacement for scipy_optim: each cost-function call issues a
        single eval_batch call with batch_size=1 instead of a Qulacs CPU call.
        No cross-env batching — the speedup comes from GPU arithmetic over CPU.
        """
        state   = self.state.clone()
        thetas  = state[:, self.num_qubits + 3:]
        rot_pos = (state[:, self.num_qubits: self.num_qubits + 3] == 1).nonzero(as_tuple=True)
        x0      = np.asarray(thetas[rot_pos].cpu().detach(), dtype=np.float32)

        if len(x0) == 0:
            return thetas, 0, x0

        bvqe   = self._batched_vqe
        bdev   = bvqe.device
        st_gpu = state.to(bdev)

        def cost(x):
            probe = torch.tensor(x.reshape(1, -1), dtype=torch.float32, device=bdev)
            return bvqe.eval_batch(st_gpu, probe).item()

        result = scipy.optimize.minimize(
            cost, x0=x0, method=method,
            options={"maxiter": self.global_iters},
        )
        thetas[rot_pos] = torch.tensor(result["x"], dtype=torch.float)
        return thetas, result["nfev"], result["x"]

    # ── Rotosolve optimiser ────────────────────────────────────────────────────

    def rotosolve_optim(self):
        """Rotosolve angle optimisation — drop-in replacement for scipy_optim.

        For each rotation parameter θ_i in sequence, evaluates E at θ_i ∈
        {0, π/2, π} with all other parameters fixed, fits the model
            E(θ_i) = A·cos(θ_i) + B·sin(θ_i) + C
        and sets θ_i to its analytical minimum atan2(−B, −A).
        Repeats for self.rotosolve_sweeps full sweeps over all parameters.

        Returns (thetas, nfev, opt_angles) — identical interface to scipy_optim.
        """
        state    = self.state.clone()
        thetas   = state[:, self.num_qubits + 3:]
        rot_pos  = (state[:, self.num_qubits: self.num_qubits + 3] == 1).nonzero(as_tuple=True)
        angles   = np.asarray(thetas[rot_pos].cpu().detach(), dtype=float)
        n_params = len(angles)

        if n_params == 0:
            return thetas, 0, angles

        bvqe     = self._batched_vqe
        bdev     = bvqe.device
        n_probes = 3 * n_params       # total probe configs per sweep

        # Probe values for Rotosolve: {0, π/2, π}
        probe_vals = np.array([0.0, np.pi / 2.0, np.pi], dtype=np.float32)
        nfev = 0

        for _ in range(self.rotosolve_sweeps):
            # Build probe angle matrix (n_probes, n_params) in one NumPy op.
            # Row 3i+p: identical to current angles except angles[i] = probe_vals[p].
            probe_matrix = np.tile(angles.astype(np.float32), (n_probes, 1))
            for i in range(n_params):
                probe_matrix[3*i : 3*i+3, i] = probe_vals

            angle_t  = torch.tensor(probe_matrix, device=bdev)        # (n_probes, n_params)
            energies = bvqe.eval_batch(state.to(bdev), angle_t).cpu().numpy()
            nfev    += n_probes

            # Analytical minimum for each parameter
            for i in range(n_params):
                e0, epi2, epi = energies[3*i], energies[3*i+1], energies[3*i+2]
                A = (e0 - epi) / 2.0
                C = (e0 + epi) / 2.0
                B = epi2 - C
                angles[i] = float(np.arctan2(-B, -A))

        thetas[rot_pos] = torch.tensor(angles, dtype=torch.float)
        return thetas, nfev, angles

    def spsa_optim(self, method="SPSA"):
        """SPSA-style optimiser with optional Adam preconditioning.

        Uses BatchedVQE (GPU) for energy evaluation — same backend as
        rotosolve_optim.  Each iteration batches one or more independent
        +/- perturbation probe pairs into one eval_batch call. The number of
        directions is chosen automatically from the number of active rotation
        parameters.

        Supported method names:
          - ``SPSA``: vanilla SPSA updates
          - ``AdamSPSA``: SPSA gradient estimate + Adam-style update

        Hyperparameters are read from ``self.options`` when present; otherwise
        conservative defaults are used.
        """
        state    = self.state.clone()
        thetas   = state[:, self.num_qubits + 3:]
        rot_pos  = (state[:, self.num_qubits: self.num_qubits + 3] == 1).nonzero(as_tuple=True)
        angles   = np.asarray(thetas[rot_pos].cpu().detach(), dtype=np.float32)
        n_params = len(angles)

        if n_params == 0:
            return thetas, 0, angles

        bvqe   = self._batched_vqe
        bdev   = bvqe.device
        st_gpu = state.to(bdev)

        def cost_batch(angle_matrix):
            """angle_matrix: (B, n_params) ndarray → (B,) ndarray of energies."""
            t = torch.tensor(angle_matrix, dtype=torch.float32, device=bdev)
            return bvqe.eval_batch(st_gpu, t).detach().cpu().numpy()

        opts   = getattr(self, "options", {})
        a      = float(opts.get("a", 0.05))
        alpha  = float(opts.get("alpha", 0.602))
        c      = float(opts.get("c", 0.1))
        gamma  = float(opts.get("gamma", 0.101))
        A      = float(opts.get("lamda", max(10.0, self.global_iters * 0.1)))
        beta_1 = float(opts.get("beta_1", 0.9))
        beta_2 = float(opts.get("beta_2", 0.999))
        spsa_batch = self._auto_spsa_batch_size(n_params)

        use_adam = str(method).upper() == "ADAMSPSA"
        x        = angles.reshape(1, -1)          # (1, n_params)
        best_val = cost_batch(x)[0]
        best_x   = x.copy()
        nfev     = 1

        m = np.zeros_like(x)
        v = np.zeros_like(x)
        iters = max(1, int(self.global_iters))

        for epoch in range(iters):
            ak    = vc._spsa_lr(epoch, a=a, A=A, alpha=alpha)
            ck    = vc._spsa_ck(epoch, c=c, gamma=gamma)
            delta = np.random.choice(
                [-1.0, 1.0], size=(spsa_batch, x.shape[1])
            ).astype(np.float32)

            probe_plus  = x + ck * delta
            probe_minus = x - ck * delta
            probe = np.concatenate([probe_plus, probe_minus], axis=0)          # (2B, n_params)
            e     = cost_batch(probe)                                           # (2B,)
            nfev += 2 * spsa_batch

            e_plus  = e[:spsa_batch].reshape(-1, 1)
            e_minus = e[spsa_batch:].reshape(-1, 1)
            grad_dirs = ((e_plus - e_minus) / (2.0 * ck)) / delta              # (B, n_params)
            grad = grad_dirs.mean(axis=0, keepdims=True)                        # (1, n_params)

            if use_adam:
                m    = beta_1 * m + (1.0 - beta_1) * grad
                v    = beta_2 * v + (1.0 - beta_2) * grad ** 2
                mhat = m / (1.0 - beta_1 ** (epoch + 1))
                vhat = v / (1.0 - beta_2 ** (epoch + 1))
                x    = x - ak * mhat / (np.sqrt(vhat) + 1e-8)
            else:
                x = x - ak * grad

            val = cost_batch(x)[0]
            nfev += 1
            if val < best_val:
                best_val = val
                best_x   = x.copy()

        best_x_flat = best_x.flatten()
        thetas[rot_pos] = torch.tensor(best_x_flat, dtype=torch.float32)
        return thetas, nfev, best_x_flat

    def psr_adam_optim(self):
        """Parameter Shift Rule + Adam optimiser.

        Computes the exact gradient via PSR — 2 evaluations per parameter,
        all batched into one eval_batch call (2*n_params rows) per Adam step.
        Total GPU calls: K + 1  (K gradient evals + 1 final energy check).

        global_iters controls the number of Adam steps K (10–20 is typically
        sufficient; contrast with SPSA which needs 100–1000).
        """
        state    = self.state.clone()
        thetas   = state[:, self.num_qubits + 3:]
        rot_pos  = (state[:, self.num_qubits: self.num_qubits + 3] == 1).nonzero(as_tuple=True)
        raw      = np.asarray(thetas[rot_pos].cpu().detach(), dtype=np.float32)
        n_params = len(raw)

        if n_params == 0:
            return thetas, 0, raw

        # Break theta=0 symmetry: PSR gradient is zero whenever all angles are
        # exactly 0 (the gradient of sin(θ) at θ=0 is non-zero, but for
        # circuits with many zero-initialised gates the landscape can be flat
        # enough that Adam makes no progress). Matches psr_adam_optim_cpu fix.
        zero_mask = np.abs(raw) < 1e-6
        raw[zero_mask] = np.random.uniform(-0.3, 0.3, int(zero_mask.sum())).astype(np.float32)
        angles = raw

        bvqe   = self._batched_vqe
        bdev   = bvqe.device
        st_gpu = state.to(bdev)

        opts   = getattr(self, "options", {})
        lr     = float(opts.get("lr", 0.01))
        beta_1 = float(opts.get("beta_1", 0.9))
        beta_2 = float(opts.get("beta_2", 0.999))
        K      = max(1, int(self.global_iters))

        x      = angles.reshape(1, -1)   # (1, n_params)
        nfev   = 0
        m      = np.zeros_like(x)
        v      = np.zeros_like(x)

        # Initial energy for best-solution tracking
        init_t   = torch.tensor(x, dtype=torch.float32, device=bdev)
        best_val = bvqe.eval_batch(st_gpu, init_t).item()
        best_x   = x.copy()
        nfev    += 1

        for step in range(K):
            # PSR probe matrix: (2*n_params, n_params)
            # Row 2i:   x with x[i] + pi/2
            # Row 2i+1: x with x[i] - pi/2
            probe = np.repeat(x, 2 * n_params, axis=0)   # (2n, n)
            for i in range(n_params):
                probe[2 * i,     i] += np.pi / 2
                probe[2 * i + 1, i] -= np.pi / 2

            probe_t  = torch.tensor(probe, dtype=torch.float32, device=bdev)
            energies = bvqe.eval_batch(st_gpu, probe_t).detach().cpu().numpy()  # (2n,)
            nfev    += 2 * n_params

            grad = (energies[0::2] - energies[1::2]) / 2.0  # (n_params,) exact
            grad = grad.reshape(1, -1)

            m    = beta_1 * m + (1 - beta_1) * grad
            v    = beta_2 * v + (1 - beta_2) * grad ** 2
            mhat = m / (1 - beta_1 ** (step + 1))
            vhat = v / (1 - beta_2 ** (step + 1))
            x    = x - lr * mhat / (np.sqrt(vhat) + 1e-8)

        # Final evaluation — keep better of initial vs optimised
        final_t   = torch.tensor(x, dtype=torch.float32, device=bdev)
        final_val = bvqe.eval_batch(st_gpu, final_t).item()
        nfev     += 1
        if final_val < best_val:
            best_x = x.copy()

        best_x_flat = best_x.flatten()
        thetas[rot_pos] = torch.tensor(best_x_flat, dtype=torch.float32)
        return thetas, nfev, best_x_flat

    # ── CPU optimisers (Qulacs-based, for GPU vs CPU benchmarking) ────────────

    def _qulacs_cost_fn(self, state):
        """Return a callable cost(angles_vec) using Qulacs CPU evaluation."""
        qulacs_inst   = vc.Parametric_Circuit(self.num_qubits, self.noise_models, self.noise_values)
        qulacs_circuit = qulacs_inst.construct_ansatz(state)
        def cost(x):
            return vc.get_energy_qulacs(
                x, observable=self.hamiltonian, weights=self.weights,
                circuit=qulacs_circuit, n_qubits=self.num_qubits,
                energy_shift=self.energy_shift, n_shots=int(self.n_shots),
                phys_noise=self.phys_noise, state=self.ket,
            )
        return cost

    def rotosolve_optim_cpu(self):
        """Rotosolve using Qulacs CPU energy evaluation (no BatchedVQE).

        Same algorithm as rotosolve_optim() but each of the 3*n_params probe
        evaluations is a sequential Qulacs CPU call.  Used for CPU vs GPU
        benchmarking — the wall-clock difference shows the GPU batching gain.
        """
        state    = self.state.clone()
        thetas   = state[:, self.num_qubits + 3:]
        rot_pos  = (state[:, self.num_qubits: self.num_qubits + 3] == 1).nonzero(as_tuple=True)
        angles   = np.asarray(thetas[rot_pos].cpu().detach(), dtype=np.float64)
        n_params = len(angles)

        if n_params == 0:
            return thetas, 0, angles

        cost     = self._qulacs_cost_fn(state)
        probe_vals = np.array([0.0, np.pi / 2.0, np.pi])
        nfev     = 0

        for _ in range(self.rotosolve_sweeps):
            for i in range(n_params):
                orig = angles[i]
                e_vals = []
                for pv in probe_vals:
                    angles[i] = pv
                    e_vals.append(cost(angles))
                    nfev += 1
                angles[i] = orig
                e0, epi2, epi = e_vals
                A = (e0 - epi) / 2.0
                C = (e0 + epi) / 2.0
                B = epi2 - C
                angles[i] = float(np.arctan2(-B, -A))

        thetas[rot_pos] = torch.tensor(angles, dtype=torch.float)
        return thetas, nfev, angles

    def psr_adam_optim_cpu(self):
        """PSR + Adam using Qulacs CPU energy evaluation (no BatchedVQE).

        Same algorithm as psr_adam_optim() but each PSR probe (2 per parameter
        per Adam step) is a sequential Qulacs CPU call.  Used for CPU vs GPU
        benchmarking.  Includes the theta=0 perturbation fix.
        """
        state    = self.state.clone()
        thetas   = state[:, self.num_qubits + 3:]
        rot_pos  = (state[:, self.num_qubits: self.num_qubits + 3] == 1).nonzero(as_tuple=True)
        raw      = np.asarray(thetas[rot_pos].cpu().detach(), dtype=np.float64)
        n_params = len(raw)

        if n_params == 0:
            return thetas, 0, raw

        # Break theta=0 symmetry (same fix as _psr_adam_envs_batched)
        zero_mask = np.abs(raw) < 1e-6
        raw[zero_mask] = np.random.uniform(-0.3, 0.3, zero_mask.sum())
        angles = raw.copy()

        opts   = getattr(self, "options", {})
        lr     = float(opts.get("lr", 0.01))
        beta_1 = float(opts.get("beta_1", 0.9))
        beta_2 = float(opts.get("beta_2", 0.999))
        K      = max(1, int(self.global_iters))

        cost     = self._qulacs_cost_fn(state)
        best_val = cost(angles)
        best_x   = angles.copy()
        nfev     = 1

        m = np.zeros(n_params)
        v = np.zeros(n_params)

        for step in range(K):
            grad = np.zeros(n_params)
            for i in range(n_params):
                x_plus          = angles.copy(); x_plus[i]  += np.pi / 2
                x_minus         = angles.copy(); x_minus[i] -= np.pi / 2
                grad[i] = (cost(x_plus) - cost(x_minus)) / 2.0
                nfev += 2

            m    = beta_1 * m + (1 - beta_1) * grad
            v    = beta_2 * v + (1 - beta_2) * grad ** 2
            mhat = m / (1 - beta_1 ** (step + 1))
            vhat = v / (1 - beta_2 ** (step + 1))
            angles = angles - lr * mhat / (np.sqrt(vhat) + 1e-8)

        final_val = cost(angles)
        nfev += 1
        if final_val < best_val:
            best_x = angles.copy()

        thetas[rot_pos] = torch.tensor(best_x, dtype=torch.float32)
        return thetas, nfev, best_x

    # ── reward function ───────────────────────────────────────────────────────

    def reward_fn(self, energy):
        fn = self.fn_type
        max_depth = self.step_counter == (self.num_layers - 1)
        prev_gap = abs(self.prev_energy - self.min_eig) + 1e-10
        start_gap = abs(self.start_energy - self.min_eig) + 1e-10

        if fn == "incremental_with_fixed_ends":
            if self.error < self.done_threshold:
                return 5.0
            elif max_depth:
                return -5.0
            else:
                return float(np.clip(
                    (self.prev_energy - energy) / prev_gap,
                    -1, 1,
                ))
        elif fn == "incremental_with_fixed_start":
            if self.error < self.done_threshold:
                return 5.0
            elif max_depth:
                return -5.0
            else:
                return float(np.clip(
                    (self.prev_energy - energy) / start_gap,
                    -1, 1,
                ))
        elif fn == "nive_fives":
            if self.error < self.done_threshold:
                return 5.0
            elif max_depth:
                return -5.0
            return 0.0
        elif fn == "incremental":
            return (self.prev_energy - energy) / prev_gap
        elif fn == "incremental_clipped":
            return float(np.clip(
                (self.prev_energy - energy) / prev_gap,
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
                    (self.prev_energy - energy) / prev_gap,
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
        illegal_decoded = [k for k, v in self.action_dict.items() if v in ill]
        self.illegal_actions = ill
        return illegal_decoded
