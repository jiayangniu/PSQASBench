"""
PSQASBench top-level utilities
"""

import argparse
import os
import sys
from pathlib import Path
import torch


# ── Path constants ────────────────────────────────────────────────────────────

ROOT       = Path(__file__).resolve().parent
MOL_DIR    = ROOT / "mol_data"
CONFIG_DIR = ROOT / "configs"
RESULT_DIR = ROOT / "results"


def resolve_results_root(path_str: str | None = None) -> Path:
    """Resolve the benchmark output root from CLI, env, or repo default."""
    raw_value = path_str or os.environ.get("PSQAS_RESULTS_DIR")
    if not raw_value:
        return RESULT_DIR

    candidate = Path(raw_value).expanduser()
    if not candidate.is_absolute():
        candidate = ROOT / candidate
    return candidate.resolve()


# ── Molecule registry ─────────────────────────────────────────────────────────

MOL_FILES = {
    "L1_H2_Equil_4q":       "L1_H2_Equil_4q_geom_H_.0_.0_0.0;_H_.0_.0_0.735_jordan_wigner.npz",
    "L1_BH_6q":             "L1_BH_6q_geom_B_.0_.0_0.0;_H_.0_.0_1.232_jordan_wigner.npz",
    "L2_BeH_Plus_4q":       "L2_BeH_Plus_4q_geom_Be_.0_.0_0.0;_H_.0_.0_1.312_jordan_wigner.npz",
    "L2_LiH_Equil_6q":      "L2_LiH_Equil_6q_geom_Li_.0_.0_0.0;_H_.0_.0_1.595_jordan_wigner.npz",
    "L2_BF_8q":             "L2_BF_8q_geom_B_.0_.0_0.0;_F_.0_.0_1.267_jordan_wigner.npz",
    "L3_HeH_Plus_4q":       "L3_HeH_Plus_4q_geom_He_.0_.0_0.0;_H_.0_.0_0.774_jordan_wigner.npz",
    "L3_CH2_Singlet_6q":    "L3_CH2_Singlet_6q_geom_C_.0_.0_0.0;_H_.0_0.86_0.73;_H_.0_-0.86_0.73_jordan_wigner.npz",
    "L3_CH2_Singlet_8q":    "L3_CH2_Singlet_8q_geom_C_.0_.0_0.0;_H_.0_0.86_0.73;_H_.0_-0.86_0.73_jordan_wigner.npz",
    "L3_CH2_Singlet_R113_A080_6q": "L3_CH2_Singlet_R113_A080_6q_geom_C_.0_.0_0.0;_H_.0_0.726_0.866;_H_.0_-0.726_0.866_jordan_wigner.npz",
    "L3_CH2_Singlet_R113_A130_6q": "L3_CH2_Singlet_R113_A130_6q_geom_C_.0_.0_0.0;_H_.0_1.024_0.478;_H_.0_-1.024_0.478_jordan_wigner.npz",
    "L3_CH2_Singlet_R130_A100_6q": "L3_CH2_Singlet_R130_A100_6q_geom_C_.0_.0_0.0;_H_.0_0.996_0.836;_H_.0_-0.996_0.836_jordan_wigner.npz",
    "L3_CH2_Singlet_R130_A130_6q": "L3_CH2_Singlet_R130_A130_6q_geom_C_.0_.0_0.0;_H_.0_1.178_0.549;_H_.0_-1.178_0.549_jordan_wigner.npz",
    "L3_CH2_Singlet_R130_A100_8q": "L3_CH2_Singlet_R130_A100_8q_geom_C_.0_.0_0.0;_H_.0_0.996_0.836;_H_.0_-0.996_0.836_jordan_wigner.npz",
    "L3_CH2_Singlet_R130_A130_8q": "L3_CH2_Singlet_R130_A130_8q_geom_C_.0_.0_0.0;_H_.0_1.178_0.549;_H_.0_-1.178_0.549_jordan_wigner.npz",
    "L3_LiH_Stretch_6q":    "L3_LiH_Stretch_6q_geom_Li_.0_.0_0.0;_H_.0_.0_3.500_jordan_wigner.npz",
    "LiH_6q_2p2":           "LiH_6q_geom_Li_.0_.0_.0;_H_.0_.0_2.2_jordan_wigner.npz",
    "L3_H3_Triangle_6q":    "L3_H3_Triangle_6q_geom_H_.0_.0_0.0;_H_1.0_.0_0.0;_H_0.5_0.866_0.0_jordan_wigner.npz",
    "L4_H2_Stretch_4q":     "L4_H2_Stretch_4q_geom_H_.0_.0_0.0;_H_.0_.0_2.5_jordan_wigner.npz",
    "L4_H3_Linear_6q":      "L4_H3_Linear_6q_geom_H_.0_.0_0.0;_H_.0_.0_1.0;_H_.0_.0_2.0_jordan_wigner.npz",
    "L4_H2O_StrongCorr_8q": "L4_H2O_StrongCorr_8q_geom_O_.0_.0_0.0;_H_.0_1.186_0.918;_H_.0_-1.186_0.918_jordan_wigner.npz",
    "L5_H3_Linear_6q":      "L5_H3_Linear_6q_geom_H_.0_.0_0.0;_H_.0_.0_1.0;_H_.0_.0_2.0_jordan_wigner.npz",
    "L5_H4_Chain_8q":       "L5_H4_Chain_8q_geom_H_.0_.0_0.0;_H_.0_.0_1.0;_H_.0_.0_2.0;_H_.0_.0_3.0_jordan_wigner.npz",
    "L6_BeH2_10q":          "L6_BeH2_Scalability_10q_geom_Be_.0_.0_0.0;_H_.0_.0_1.326;_H_.0_.0_-1.326_jordan_wigner.npz",
    # BeH2 basis-set ladder: 6q STO-3G now serves as the L1 anchor.
    "L1_BeH2_STO3G_6q":    "L1_BeH2_STO3G_6q_geom_Be_.0_.0_0.0;_H_.0_.0_1.326;_H_.0_.0_-1.326_jordan_wigner.npz",
    "L6_BeH2_631G_8q":     "L6_BeH2_631G_8q_geom_Be_.0_.0_0.0;_H_.0_.0_1.326;_H_.0_.0_-1.326_jordan_wigner.npz",
    "L6_BeH2_6311G_10q":   "L6_BeH2_6311G_10q_geom_Be_.0_.0_0.0;_H_.0_.0_1.326;_H_.0_.0_-1.326_jordan_wigner.npz",
    "L6_BeH2_CCPVDZ_12q":  "L6_BeH2_CCPVDZ_12q_geom_Be_.0_.0_0.0;_H_.0_.0_1.326;_H_.0_.0_-1.326_jordan_wigner.npz",
    "L6_BeH2_CCPVDZ_14q":  "L6_BeH2_CCPVDZ_14q_geom_Be_.0_.0_0.0;_H_.0_.0_1.326;_H_.0_.0_-1.326_jordan_wigner.npz",
}


# ── Stdout redirection ────────────────────────────────────────────────────────

def redirect_output(file_path: Path):
    """Redirect both stdout and stderr to a file. Returns the file object for later closing."""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    log = open(file_path, "w", buffering=1, encoding="utf-8")
    sys.stdout = log
    sys.stderr = log
    return log



# ── Argument parsing ──────────────────────────────────────────────────────────

METHODS = ["crlqas", "hyrlqas", "qdarts", "tfqas", "gqeqas"]
METHOD_CONFIG_DIR = {
    "crlqas":  "crlqas",
    "hyrlqas": "hyrlqas",
    "qdarts":  "qdarts",
    "tfqas":   "tfqas",
    "gqeqas":  "gqe",
}


def parse_args() -> dict:
    """
    Parse command-line arguments.  All four flags are required so the script
    can be launched non-interactively (nohup, screen, SLURM, etc.).

    Usage:
        python main.py --method crlqas  --mol L1_H2_Equil_4q --seed 11111 --device cuda:0
        python main.py --method hyrlqas --mol L1_H2_Equil_4q --seed 11111 --device cuda:0
        python main.py --method crlqas --mol LiH_6q_2p2 --config LiH_6q_2p2_cobyla.cfg --seed 11111 --device cuda:0
    """
    MOL_KEYS = list(MOL_FILES.keys())

    parser = argparse.ArgumentParser(
        prog="PSQASBench",
        description="Principled benchmark for RL-based Quantum Architecture Search.",
    )
    parser.add_argument("--method", required=True, choices=METHODS,
                        help=f"QAS method to run. Choices: {METHODS}")
    parser.add_argument("--mol",    required=True, choices=MOL_KEYS,
                        help="Molecule key (see bench_utils.MOL_FILES)")
    parser.add_argument("--config", required=False,
                        help="Config filename inside configs/<method>/; defaults to <mol>.cfg")
    parser.add_argument("--seed",   required=True, type=int,
                        help="Random seed (e.g. 11111)")
    parser.add_argument("--device", required=True,
                        help="Torch device string (e.g. cuda:0, cpu)")
    parser.add_argument("--save-summary-detailed", required=False, type=int, choices=[0, 1],
                        help="Whether to save the legacy detailed summary_<seed>.npy (0/1)")
    parser.add_argument("--use-wandb", required=False, type=int, choices=[0, 1],
                        help="Whether to enable Weights & Biases logging/upload (0/1)")
    parser.add_argument("--results-root", required=False,
                        help="Output root for run artifacts. Defaults to $PSQAS_RESULTS_DIR or ./results")

    args = parser.parse_args()
    return dict(
        method=args.method,
        mol=args.mol,
        seed=args.seed,
        device=args.device,
        config=(args.config if args.config else f"{args.mol}.cfg"),
        save_summary_detailed=args.save_summary_detailed,
        use_wandb=args.use_wandb,
        results_root=resolve_results_root(args.results_root),
    )


# ── Runner factory ────────────────────────────────────────────────────────────

def get_runner(method: str, config_path: Path, mol_path: Path,
               result_dir: Path, seed: int, device: torch.device,
               mol_name: str | None = None,
               save_summary_detailed: int | None = None,
               use_wandb: int | None = None):
    if method == "crlqas":
        from RLQAS import CRLQASRunner
        return CRLQASRunner(
            config_path,
            mol_path,
            result_dir,
            seed,
            device,
            mol_name=mol_name,
            save_summary_detailed=save_summary_detailed,
            use_wandb=use_wandb,
        )
    if method == "hyrlqas":
        from RLQAS import HyRLQASRunner
        return HyRLQASRunner(
            config_path,
            mol_path,
            result_dir,
            seed,
            device,
            mol_name=mol_name,
            save_summary_detailed=save_summary_detailed,
            use_wandb=use_wandb,
        )
    if method == "qdarts":
        from QuantumDARTS import QuantumDARTSRunner
        return QuantumDARTSRunner(
            config_path,
            mol_path,
            result_dir,
            seed,
            device,
        )
    if method == "tfqas":
        from TFQAS import TFQASRunner
        return TFQASRunner(config_path, mol_path, result_dir, seed, device)
    if method == "gqeqas":
        from GQEQAS import GQERunner
        return GQERunner(config_path, mol_path, result_dir, seed, device)
    raise ValueError(f"Unknown method '{method}'. Implemented so far: {METHODS}")
