"""
Base class for all QAS method runners in PSQASBench.
Each method implements run() and returns a standardized result dict.
"""
from abc import ABC, abstractmethod
from pathlib import Path
import numpy as np


class BaseRunner(ABC):
    """Abstract base for all benchmark runners."""

    def __init__(self, config, mol_path: Path, result_dir: Path, seed: int):
        """
        Args:
            config:     Parsed config object (configparser.ConfigParser)
            mol_path:   Path to the .npz Hamiltonian file
            result_dir: Directory to save results
            seed:       Random seed for this run
        """
        self.config = config
        self.mol_path = mol_path
        self.result_dir = result_dir
        self.seed = seed

        result_dir.mkdir(parents=True, exist_ok=True)
        self._load_molecule()

    def _load_molecule(self):
        """Load Hamiltonian and exact ground state energy from .npz."""
        data = np.load(self.mol_path, allow_pickle=True)
        self.hamiltonian = data["hamiltonian"]
        self.weights = data["weights"]
        self.exact_energy = float(data["eigvals"][0])  # ground state
        self.energy_shift = float(data.get("energy_shift", 0.0))

    @abstractmethod
    def run(self) -> dict:
        """
        Execute the QAS experiment.

        Returns a dict with standardized keys:
            best_energy      (float)  : lowest energy found
            energy_error     (float)  : |best_energy - exact_energy| in Ha
            cnot_count       (int)    : CNOT gates in best circuit
            circuit_depth    (int)    : depth of best circuit
            nfev             (int)    : total VQE function evaluations
            success          (bool)   : energy_error < accept_err (1.6 mHa)
            energy_history   (list)   : energy per episode
            cnot_history     (list)   : CNOT count per episode
            seed             (int)    : random seed used
        """
        raise NotImplementedError

    def save_result(self, result: dict):
        """Save result dict as .npz in result_dir."""
        out_path = self.result_dir / f"result_seed{self.seed}.npz"
        np.savez(out_path, **{k: np.array(v) for k, v in result.items()})
        return out_path
