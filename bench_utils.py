"""PSQASBench top-level utilities."""

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

# Final benchmark-facing molecule keys.  These map onto the existing mol_data
# artefacts, whose filenames intentionally remain unchanged for compatibility.
FINAL_SUITE_MOL_FILES = {
    "T1_BeH2_STO3G_6q":    MOL_FILES["L1_BeH2_STO3G_6q"],
    "T1_LiH_Equil_6q":     MOL_FILES["L2_LiH_Equil_6q"],
    "T2_CH2_8q":           MOL_FILES["L3_CH2_Singlet_8q"],
    "T3_H2_Stretch_4q":    MOL_FILES["L4_H2_Stretch_4q"],
    "T3_H2O_StrongCorr_8q": MOL_FILES["L4_H2O_StrongCorr_8q"],
    "T3_H4_Chain_8q":      MOL_FILES["L5_H4_Chain_8q"],
    "T4_H3_Linear_6q":     MOL_FILES["L5_H3_Linear_6q"],
    "T5_BeH2_631G_8q":     MOL_FILES["L6_BeH2_631G_8q"],
    "T5_BeH2_6311G_10q":   MOL_FILES["L6_BeH2_6311G_10q"],
    "T5_BeH2_CCPVDZ_12q":  MOL_FILES["L6_BeH2_CCPVDZ_12q"],
    "T5_BeH2_CCPVDZ_14q":  MOL_FILES["L6_BeH2_CCPVDZ_14q"],
}
MOL_FILES.update(FINAL_SUITE_MOL_FILES)

LEGACY_BENCH_ALIASES = {
    "L1_BeH2_STO3G_6q": "T1_BeH2_STO3G_6q",
    "L1_LiH_Equil_6q": "T1_LiH_Equil_6q",
    "L2_LiH_Equil_6q": "T1_LiH_Equil_6q",
    "L3_CH2_Singlet_8q": "T2_CH2_8q",
    "L4_H2_Stretch_4q": "T3_H2_Stretch_4q",
    "L4_H2O_StrongCorr_8q": "T3_H2O_StrongCorr_8q",
    "L5_H4_Chain_8q": "T3_H4_Chain_8q",
    "L4_H3_Linear_6q": "T4_H3_Linear_6q",
    "L5_H3_Linear_6q": "T4_H3_Linear_6q",
    "L6_BeH2_631G_8q": "T5_BeH2_631G_8q",
    "L6_BeH2_6311G_10q": "T5_BeH2_6311G_10q",
    "L6_BeH2_CCPVDZ_12q": "T5_BeH2_CCPVDZ_12q",
    "L6_BeH2_CCPVDZ_14q": "T5_BeH2_CCPVDZ_14q",
}

CONFIG_NAME_REWRITES = (
    ("L1_BeH2_STO3G_6q", "T1_BeH2_STO3G_6q"),
    ("L1_LiH_Equil_6q", "T1_LiH_Equil_6q"),
    ("L2_LiH_Equil_6q", "T1_LiH_Equil_6q"),
    ("L3_CH2_Singlet_8q", "T2_CH2_8q"),
    ("L4_H2_Stretch_4q", "T3_H2_Stretch_4q"),
    ("L4_H2O_StrongCorr_8q", "T3_H2O_StrongCorr_8q"),
    ("L5_H4_Chain_8q", "T3_H4_Chain_8q"),
    ("L4_H3_Linear_6q", "T4_H3_Linear_6q"),
    ("L5_H3_Linear_6q", "T4_H3_Linear_6q"),
    ("L6_BeH2_631G_8q", "T5_BeH2_631G_8q"),
    ("L6_BeH2_6311G_10q", "T5_BeH2_6311G_10q"),
    ("L6_BeH2_CCPVDZ_12q", "T5_BeH2_CCPVDZ_12q"),
    ("L6_BeH2_CCPVDZ_14q", "T5_BeH2_CCPVDZ_14q"),
)

METHOD_CONFIG_ALIASES = {
    "crlqas": {
        "Formal_EXP/L1_LiH_Equil_6q_cobyla_20k_depth10.cfg": "Formal_EXP/T1_LiH_Equil_6q_cobyla_20k_depth10.cfg",
        "Formal_EXP/T1_LiH_Equil_6q_cobyla_20k_curriculum.cfg": "Formal_EXP/T1_LiH_Equil_6q_cobyla_20k_depth50.cfg",
        "Formal_EXP/L2_LiH_Equil_6q_cobyla_20k_curriculum.cfg": "Formal_EXP/T1_LiH_Equil_6q_cobyla_20k_depth50.cfg",
    },
    "hyrlqas": {
        "Formal_EXP/L1_LiH_Equil_6q_cobyla_20k_depth10.cfg": "Formal_EXP/T1_LiH_Equil_6q_cobyla_20k_depth10.cfg",
        "Formal_EXP/T1_LiH_Equil_6q_cobyla_20k_curriculum.cfg": "Formal_EXP/T1_LiH_Equil_6q_cobyla_20k_depth50.cfg",
        "Formal_EXP/L2_LiH_Equil_6q_cobyla_20k_curriculum.cfg": "Formal_EXP/T1_LiH_Equil_6q_cobyla_20k_depth50.cfg",
    },
    "qdarts": {
        "Formal_EXP/L1_LiH_Equil_6q.cfg": "Formal_EXP/T1_LiH_Equil_6q_depth10.cfg",
        "Formal_EXP/T1_LiH_Equil_6q.cfg": "Formal_EXP/T1_LiH_Equil_6q_depth10.cfg",
        "Formal_EXP/L2_LiH_Equil_6q.cfg": "Formal_EXP/T1_LiH_Equil_6q_depth10.cfg",
    },
    "tfqas": {
        "Formal_EXP/L1_LiH_Equil_6q_depth10.cfg": "Formal_EXP/T1_LiH_Equil_6q_depth10.cfg",
        "Formal_EXP/T1_LiH_Equil_6q.cfg": "Formal_EXP/T1_LiH_Equil_6q_depth50.cfg",
        "Formal_EXP/L2_LiH_Equil_6q.cfg": "Formal_EXP/T1_LiH_Equil_6q_depth50.cfg",
    },
    "gqeqas": {
        "Formal_EXP/L1_LiH_Equil_6q_depth10.cfg": "Formal_EXP/T1_LiH_Equil_6q_depth10.cfg",
        "Formal_EXP/T1_LiH_Equil_6q.cfg": "Formal_EXP/T1_LiH_Equil_6q_depth50.cfg",
        "Formal_EXP/L2_LiH_Equil_6q.cfg": "Formal_EXP/T1_LiH_Equil_6q_depth50.cfg",
    },
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

METHOD_DEFAULT_CONFIGS = {
    "crlqas": {
        "T1_BeH2_STO3G_6q": "Formal_EXP/T1_BeH2_STO3G_6q_cobyla_20k_depth10.cfg",
        "T1_LiH_Equil_6q": "Formal_EXP/T1_LiH_Equil_6q_cobyla_20k_depth50.cfg",
        "T2_CH2_8q": "Formal_EXP/T2_CH2_8q_rotosolve_s2_20k.cfg",
        "T3_H2_Stretch_4q": "Formal_EXP/T3_H2_Stretch_4q_cobyla_20k.cfg",
        "T3_H2O_StrongCorr_8q": "Formal_EXP/T3_H2O_StrongCorr_8q_rotosolve_s2_20k.cfg",
        "T3_H4_Chain_8q": "Formal_EXP/T3_H4_Chain_8q_rotosolve_s2_20k_all.cfg",
        "T4_H3_Linear_6q": "Formal_EXP/T4_H3_Linear_6q_cobyla_20k_linear.cfg",
        "T5_BeH2_631G_8q": "Formal_EXP/T5_BeH2_631G_8q_rotosolve_s2_20k.cfg",
        "T5_BeH2_6311G_10q": "Formal_EXP/T5_BeH2_6311G_10q_rotosolve_s2_20k.cfg",
        "T5_BeH2_CCPVDZ_12q": "Formal_EXP/T5_BeH2_CCPVDZ_12q_rotosolve_s2_20k.cfg",
        "T5_BeH2_CCPVDZ_14q": "Formal_EXP/T5_BeH2_CCPVDZ_14q_rotosolve_s2_20k.cfg",
    },
    "hyrlqas": {
        "T1_BeH2_STO3G_6q": "Formal_EXP/T1_BeH2_STO3G_6q_cobyla_20k_depth10.cfg",
        "T1_LiH_Equil_6q": "Formal_EXP/T1_LiH_Equil_6q_cobyla_20k_depth50.cfg",
        "T2_CH2_8q": "Formal_EXP/T2_CH2_8q_rotosolve_s2_20k.cfg",
        "T3_H2_Stretch_4q": "Formal_EXP/T3_H2_Stretch_4q_cobyla_20k.cfg",
        "T3_H2O_StrongCorr_8q": "Formal_EXP/T3_H2O_StrongCorr_8q_rotosolve_s2_20k.cfg",
        "T3_H4_Chain_8q": "Formal_EXP/T3_H4_Chain_8q_rotosolve_s2_20k_all.cfg",
        "T4_H3_Linear_6q": "Formal_EXP/T4_H3_Linear_6q_rotosolve_s2_20k_linear.cfg",
        "T5_BeH2_631G_8q": "Formal_EXP/T5_BeH2_631G_8q_rotosolve_s2_20k.cfg",
        "T5_BeH2_6311G_10q": "Formal_EXP/T5_BeH2_6311G_10q_rotosolve_s2_20k.cfg",
        "T5_BeH2_CCPVDZ_12q": "Formal_EXP/T5_BeH2_CCPVDZ_12q_rotosolve_s2_20k.cfg",
        "T5_BeH2_CCPVDZ_14q": "Formal_EXP/T5_BeH2_CCPVDZ_14q_rotosolve_s2_20k.cfg",
    },
    "qdarts": {
        "T1_BeH2_STO3G_6q": "Formal_EXP/T1_BeH2_STO3G_6q.cfg",
        "T1_LiH_Equil_6q": "Formal_EXP/T1_LiH_Equil_6q_depth10.cfg",
        "T2_CH2_8q": "Formal_EXP/T2_CH2_8q.cfg",
        "T3_H2_Stretch_4q": "Formal_EXP/T3_H2_Stretch_4q.cfg",
        "T3_H2O_StrongCorr_8q": "Formal_EXP/T3_H2O_StrongCorr_8q.cfg",
        "T3_H4_Chain_8q": "Formal_EXP/T3_H4_Chain_8q_all.cfg",
        "T4_H3_Linear_6q": "Formal_EXP/T4_H3_Linear_6q_linear.cfg",
        "T5_BeH2_631G_8q": "Formal_EXP/T5_BeH2_631G_8q.cfg",
        "T5_BeH2_6311G_10q": "Formal_EXP/T5_BeH2_6311G_10q.cfg",
        "T5_BeH2_CCPVDZ_12q": "Formal_EXP/T5_BeH2_CCPVDZ_12q.cfg",
        "T5_BeH2_CCPVDZ_14q": "Formal_EXP/T5_BeH2_CCPVDZ_14q.cfg",
    },
    "tfqas": {
        "T1_BeH2_STO3G_6q": "Formal_EXP/T1_BeH2_STO3G_6q_depth10.cfg",
        "T1_LiH_Equil_6q": "Formal_EXP/T1_LiH_Equil_6q_depth50.cfg",
        "T2_CH2_8q": "Formal_EXP/T2_CH2_8q.cfg",
        "T3_H2_Stretch_4q": "Formal_EXP/T3_H2_Stretch_4q.cfg",
        "T3_H2O_StrongCorr_8q": "Formal_EXP/T3_H2O_StrongCorr_8q.cfg",
        "T3_H4_Chain_8q": "Formal_EXP/T3_H4_Chain_8q_all.cfg",
        "T4_H3_Linear_6q": "Formal_EXP/T4_H3_Linear_6q_linear.cfg",
        "T5_BeH2_631G_8q": "Formal_EXP/T5_BeH2_631G_8q.cfg",
        "T5_BeH2_6311G_10q": "Formal_EXP/T5_BeH2_6311G_10q.cfg",
        "T5_BeH2_CCPVDZ_12q": "Formal_EXP/T5_BeH2_CCPVDZ_12q.cfg",
        "T5_BeH2_CCPVDZ_14q": "Formal_EXP/T5_BeH2_CCPVDZ_14q.cfg",
    },
    "gqeqas": {
        "T1_BeH2_STO3G_6q": "Formal_EXP/T1_BeH2_STO3G_6q_depth10.cfg",
        "T1_LiH_Equil_6q": "Formal_EXP/T1_LiH_Equil_6q_depth50.cfg",
        "T2_CH2_8q": "Formal_EXP/T2_CH2_8q.cfg",
        "T3_H2_Stretch_4q": "Formal_EXP/T3_H2_Stretch_4q.cfg",
        "T3_H2O_StrongCorr_8q": "Formal_EXP/T3_H2O_StrongCorr_8q.cfg",
        "T3_H4_Chain_8q": "Formal_EXP/T3_H4_Chain_8q_all.cfg",
        "T4_H3_Linear_6q": "Formal_EXP/T4_H3_Linear_6q_linear.cfg",
        "T5_BeH2_631G_8q": "Formal_EXP/T5_BeH2_631G_8q.cfg",
        "T5_BeH2_6311G_10q": "Formal_EXP/T5_BeH2_6311G_10q.cfg",
        "T5_BeH2_CCPVDZ_12q": "Formal_EXP/T5_BeH2_CCPVDZ_12q.cfg",
        "T5_BeH2_CCPVDZ_14q": "Formal_EXP/T5_BeH2_CCPVDZ_14q.cfg",
    },
}


def canonicalize_mol_key(mol_key: str) -> str:
    return LEGACY_BENCH_ALIASES.get(mol_key, mol_key)


def rewrite_config_relpath(config_rel: str) -> str:
    rewritten = config_rel
    for legacy, canonical in CONFIG_NAME_REWRITES:
        rewritten = rewritten.replace(legacy, canonical)
    return rewritten


def apply_method_config_alias(method: str, config_rel: str) -> str:
    aliases = METHOD_CONFIG_ALIASES.get(method, {})
    return aliases.get(config_rel, config_rel)


def resolve_default_config(method: str, mol_key: str, config_arg: str | None) -> str:
    if config_arg:
        rewritten = rewrite_config_relpath(config_arg)
        return apply_method_config_alias(method, rewritten)
    default_configs = METHOD_DEFAULT_CONFIGS.get(method, {})
    if mol_key in default_configs:
        return default_configs[mol_key]
    return f"{mol_key}.cfg"


def parse_args() -> dict:
    """
    Parse command-line arguments.  All four flags are required so the script
    can be launched non-interactively (nohup, screen, SLURM, etc.).

    Usage:
        python main.py --method crlqas  --mol T1_BeH2_STO3G_6q --seed 11111 --device cuda:0
        python main.py --method hyrlqas --mol T2_CH2_8q --seed 11111 --device cuda:0
        python main.py --method crlqas --mol T5_BeH2_CCPVDZ_14q --config Optimizer_EXP/bench_14q_rotosolve_gpu_k10.cfg --seed 11111 --device cuda:0
    """
    mol_choices = sorted(set(MOL_FILES.keys()) | set(LEGACY_BENCH_ALIASES.keys()))

    parser = argparse.ArgumentParser(
        prog="PSQASBench",
        description="Principled benchmark for RL-based Quantum Architecture Search.",
    )
    parser.add_argument("--method", required=True, choices=METHODS,
                        help=f"QAS method to run. Choices: {METHODS}")
    parser.add_argument("--mol",    required=True, choices=mol_choices,
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
                        help="Whether to enable Weights & Biases logging/upload (0/1, default: disabled)")
    parser.add_argument("--results-root", required=False,
                        help="Output root for run artifacts. Defaults to $PSQAS_RESULTS_DIR or ./results")

    args = parser.parse_args()
    canonical_mol = canonicalize_mol_key(args.mol)
    return dict(
        method=args.method,
        mol=canonical_mol,
        seed=args.seed,
        device=args.device,
        config=resolve_default_config(args.method, canonical_mol, args.config),
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
