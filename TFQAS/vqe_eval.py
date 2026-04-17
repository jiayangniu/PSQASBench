"""
vqe_eval.py — Ground-truth VQE evaluation using COBYLA.

Loads molecular data from .npz format and optimises circuit parameters
to find the lowest energy.

.npz format:
    hamiltonian  (2^n × 2^n complex matrix) — electronic Hamiltonian
    eigvals      (2^n,)                      — exact eigenvalues (no shift)
    energy_shift scalar                      — nuclear repulsion energy
    Total exact ground state = eigvals[0] + energy_shift
    Total VQE energy         = <ψ|H|ψ> + energy_shift
"""

import numpy as np
from scipy.optimize import minimize
from dataclasses import dataclass

from .circuit import Circuit


@dataclass
class VQEResult:
    energy: float          # total energy = <ψ|H|ψ> + energy_shift
    error: float           # |energy - E_exact|  (Ha)
    error_mha: float       # error in milli-Hartree
    n_fev: int             # number of function evaluations
    params: np.ndarray     # optimised parameters
    n_cnot: int            # CNOT count of the circuit
    reached_chem: bool     # whether error < 1.6 mHa


CHEM_ACCURACY_HA = 1.6e-3  # 1.6 mHa


def load_molecule(npz_path: str) -> dict:
    """
    Load molecular data from a .npz file.

    Returns dict with keys:
        H            — Hamiltonian matrix (numpy complex128)
        energy_shift — nuclear repulsion energy (float)
        E_exact      — ground state energy including shift (float)
    """
    data = np.load(npz_path, allow_pickle=True)
    H = data['hamiltonian'].astype(np.complex128)
    energy_shift = float(data['energy_shift'])
    # eigvals may be unsorted in some data files; use the minimum to be safe.
    E_exact = float(np.min(data['eigvals'])) + energy_shift
    return {'H': H, 'energy_shift': energy_shift, 'E_exact': E_exact}


def evaluate_circuit(
    circuit: Circuit,
    mol: dict,
    n_restarts: int = 3,
    cobyla_maxiter: int = 1000,
    rng: np.random.Generator | None = None,
) -> VQEResult:
    """
    Optimise circuit parameters with COBYLA and return VQE result.

    Runs n_restarts independent COBYLA optimisations from random initialisations
    and returns the result with the lowest energy.

    Args:
        circuit:       Circuit to evaluate
        mol:           dict from load_molecule()
        n_restarts:    number of random restarts
        cobyla_maxiter: COBYLA max iterations per restart
        rng:           numpy random generator

    Returns:
        VQEResult with the best (lowest energy) result
    """
    if rng is None:
        rng = np.random.default_rng()

    H = mol['H']
    energy_shift = mol['energy_shift']
    E_exact = mol['E_exact']

    # Trivial case: no parameters
    if circuit.n_params == 0:
        elec = circuit.energy(np.array([]), H)
        total = elec + energy_shift
        err = abs(total - E_exact)
        return VQEResult(
            energy=total,
            error=err,
            error_mha=err * 1000,
            n_fev=0,
            params=np.array([]),
            n_cnot=circuit.n_cnot,
            reached_chem=err < CHEM_ACCURACY_HA,
        )

    best_energy = np.inf
    best_params = None
    total_nfev = 0

    for _ in range(n_restarts):
        x0 = rng.uniform(-np.pi, np.pi, circuit.n_params)
        nfev_count = [0]

        def cost(params):
            nfev_count[0] += 1
            return circuit.energy(params, H)

        res = minimize(
            cost,
            x0,
            method='COBYLA',
            options={'maxiter': cobyla_maxiter, 'rhobeg': 1.0},
        )
        total_nfev += nfev_count[0]

        if res.fun < best_energy:
            best_energy = res.fun
            best_params = res.x

    total_energy = best_energy + energy_shift
    err = abs(total_energy - E_exact)

    return VQEResult(
        energy=total_energy,
        error=err,
        error_mha=err * 1000,
        n_fev=total_nfev,
        params=best_params,
        n_cnot=circuit.n_cnot,
        reached_chem=err < CHEM_ACCURACY_HA,
    )
