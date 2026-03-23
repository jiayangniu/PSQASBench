"""
PSQASBench top-level utilities
"""

import sys
from pathlib import Path
import torch


# ── Path constants ────────────────────────────────────────────────────────────

ROOT       = Path(__file__).resolve().parent
MOL_DIR    = ROOT / "mol_data"
CONFIG_DIR = ROOT / "configs"
RESULT_DIR = ROOT / "results"


# ── Molecule registry ─────────────────────────────────────────────────────────

MOL_FILES = {
    "L1_H2_Equil_4q":       "L1_H2_Equil_4q_geom_H_.0_.0_0.0;_H_.0_.0_0.735_jordan_wigner.npz",
    "L1_BH_6q":             "L1_BH_6q_geom_B_.0_.0_0.0;_H_.0_.0_1.232_jordan_wigner.npz",
    "L2_BeH_Plus_4q":       "L2_BeH_Plus_4q_geom_Be_.0_.0_0.0;_H_.0_.0_1.312_jordan_wigner.npz",
    "L2_LiH_Equil_6q":      "L2_LiH_Equil_6q_geom_Li_.0_.0_0.0;_H_.0_.0_1.595_jordan_wigner.npz",
    "L2_BF_8q":             "L2_BF_8q_geom_B_.0_.0_0.0;_F_.0_.0_1.267_jordan_wigner.npz",
    "L3_HeH_Plus_4q":       "L3_HeH_Plus_4q_geom_He_.0_.0_0.0;_H_.0_.0_0.774_jordan_wigner.npz",
    "L3_CH2_Singlet_6q":    "L3_CH2_Singlet_6q_geom_C_.0_.0_0.0;_H_.0_0.86_0.73;_H_.0_-0.86_0.73_jordan_wigner.npz",
    "L3_LiH_Stretch_6q":    "L3_LiH_Stretch_6q_geom_Li_.0_.0_0.0;_H_.0_.0_3.500_jordan_wigner.npz",
    "L3_H3_Triangle_6q":    "L3_H3_Triangle_6q_geom_H_.0_.0_0.0;_H_1.0_.0_0.0;_H_0.5_0.866_0.0_jordan_wigner.npz",
    "L4_H2_Stretch_4q":     "L4_H2_Stretch_4q_geom_H_.0_.0_0.0;_H_.0_.0_2.5_jordan_wigner.npz",
    "L4_H3_Linear_6q":      "L4_H3_Linear_6q_geom_H_.0_.0_0.0;_H_.0_.0_1.0;_H_.0_.0_2.0_jordan_wigner.npz",
    "L4_H2O_StrongCorr_8q": "L4_H2O_StrongCorr_8q_geom_O_.0_.0_0.0;_H_.0_0.757_0.586;_H_.0_-0.757_0.586_jordan_wigner.npz",
    "L5_H3_Linear_6q":      "L5_H3_Linear_6q_geom_H_.0_.0_0.0;_H_.0_.0_1.0;_H_.0_.0_2.0_jordan_wigner.npz",
    "L5_H4_Chain_8q":       "L5_H4_Chain_8q_geom_H_.0_.0_0.0;_H_.0_.0_1.0;_H_.0_.0_2.0;_H_.0_.0_3.0_jordan_wigner.npz",
    "L6_BeH2_10q":          "L6_BeH2_Scalability_10q_geom_Be_.0_.0_0.0;_H_.0_.0_1.326;_H_.0_.0_-1.326_jordan_wigner.npz",
}


# ── Stdout redirection ────────────────────────────────────────────────────────

def redirect_stdout(file_path: Path):
    """Redirect stdout to a file. Returns the file object for later closing."""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    log = open(file_path, "w", buffering=1, encoding="utf-8")
    sys.stdout = log
    return log


# ── Interactive prompts ───────────────────────────────────────────────────────

def choose(prompt: str, options: list | None = None, default_idx: int = 0) -> str:
    """Unified interactive input.

    - options is not None: display a numbered menu and return the selected value.
    - options is None: read a free-form line and return the raw string (may be empty).
    """
    print(f"\n{prompt}")
    if options is None:
        return input("> ").strip()

    for i, opt in enumerate(options):
        marker = " #" if i == default_idx else ""
        print(f"  [{i}] {opt}{marker}")
    while True:
        raw = input(f"Select [default {default_idx}]: ").strip()
        if raw == "":
            return options[default_idx]
        if raw.isdigit() and 0 <= int(raw) < len(options):
            return options[int(raw)]
        print(f"  Please enter a number between 0 and {len(options) - 1}")


def prompt_args() -> dict:
    """Collect all interactive inputs and return an args dict."""
    METHODS  = ["crlqas"]
    MOL_KEYS = list(MOL_FILES.keys())
    GPU_OPTS = [f"cuda:{i}" for i in range(torch.cuda.device_count())]

    method = choose("Select method", METHODS)
    mol    = choose("Select mol",    MOL_KEYS)

    raw_seeds = choose("Seeds (space-separated integers, e.g. 11111 22222; default 11111)")
    try:
        seeds = [int(s) for s in raw_seeds.split()] if raw_seeds else [11111]
    except ValueError:
        print("  Invalid format, using default seed 11111")
        seeds = [11111]

    device = choose("Select GPU", GPU_OPTS)

    config_name = f"{mol}.cfg"
    return dict(method=method, mol=mol, seeds=seeds, device=device,
                config=config_name)


# ── Runner factory ────────────────────────────────────────────────────────────

def get_runner(method: str, config_path: Path, mol_path: Path,
               result_dir: Path, seed: int, device: torch.device):
    if method == "crlqas":
        from RLQAS import CRLQASRunner
        return CRLQASRunner(config_path, mol_path, result_dir, seed, device)
    raise ValueError(f"Unknown method '{method}'. Implemented so far: crlqas")
