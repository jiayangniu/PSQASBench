# PSQASBench

**Pauli String Quantum Architecture Search Benchmark**

A unified benchmarking framework for Reinforcement Learning–based Quantum Architecture Search (QAS).
PSQASBench exposes systematic flaws in existing RL-for-QAS methods through a standardised 6-tier molecular test suite, unified evaluation metrics, and reproducible experimental protocols.

---

## Motivation

Current QAS papers use different molecules, different circuit-quality metrics, and inconsistent checkpoint selection strategies — making cross-method comparison meaningless.  PSQASBench fixes this by providing:

- **One molecular test suite** covering 6 diagnostic tiers (L1–L6)
- **One evaluation protocol** with shared SR, CNOT@chem, D\_struct, D\_func metrics
- **One runner interface** so every method runs through the same `main.py` entry point
- **One post-hoc structure analysis tool** (`critical_structure_tool`) for diagnosing what circuit motifs RL methods actually learn

---

## Molecular Test Suite (6-Tier Diagnostic)

All Hamiltonians are Jordan-Wigner encoded and stored in `mol_data/` as `.npz` files containing `hamiltonian`, `weights`, `eigvals`, and `energy_shift`.  Chemical accuracy threshold: **1.6 mHa**.

| Tier | Difficulty Source | Molecules | Qubits | Diagnostic Target |
|------|------------------|-----------|--------|-------------------|
| L1 | Basic optimisation | H₂ (equil.), BH, BeH₂ (STO-3G) | 4, 6, 6 | **Minimalism** — can the policy prune redundant gates? |
| L2 | Asymmetry / interaction hubs | BeH⁺, LiH (equil.), BF | 4, 6, 8 | **Asymmetry** — non-uniform qubit importance |
| L3 | Near-degenerate (small gap) | HeH⁺, CH₂, LiH (stretch), H₃ (triangle) | 4, 6–8, 6, 6 | **Stability** — flat landscape with ΔE ≈ 0 |
| L4 | Strong correlation | H₂ (stretch), H₃ (linear), H₂O | 4, 6, 8 | **Representation** — high-order Pauli terms dominate |
| L5 | Topology routing | H₃ (linear), H₄ (chain) | 6, 8 | **Topology** — 1D nearest-neighbour connectivity constraint |
| L6 | Scalability | BeH₂ (basis-set ladder: STO-3G → 6-311G) | 6–14 | **Scalability** — exponential action-space growth |

> **L5 note:** action space is restricted to nearest-neighbour entangling gates; all circuits must respect 1D linear connectivity.  Config files for L5 must set `connectivity = linear` under `[env]`.

---

## Benchmark Metrics

### Primary: 2D Pareto View

| Axis | Metric | Definition |
|------|--------|------------|
| Quality | Energy Error (mHa) | \|E\_found − E\_exact\| |
| Cost | CNOT Count | Number of CNOT gates in the found circuit |

A method that reaches chemical accuracy with fewer CNOTs dominates one that uses more gates for the same quality.

### Secondary Metrics

| Metric | Definition |
|--------|------------|
| SR@chem | Fraction of K stochastic rollouts reaching chemical accuracy |
| CNOT@chem | Median CNOT count among successful rollouts |
| best\_error\_mha | Minimum energy error across all rollouts (mHa) |
| nfev@chem | VQE function evaluations until first chemical accuracy hit |

### Policy Circuit Diversity (PCD) — *new in this work*

Computed from K stochastic rollouts with fixed policy, comparing prepared states via infidelity:

```text
d(ψᵢ, ψⱼ) = 1 − |⟨ψᵢ|ψⱼ⟩|²
```

| Metric | How computed | Interpretation |
|--------|-------------|----------------|
| D\_struct | All rotation angles fixed to π/4, compare ψᵢ(π/4) | Circuit-structure diversity |
| D\_func | Optimised angles θ\*, compare ψᵢ(θ\*) | Functional-state diversity |

#### Diagnostic matrix

| D\_struct | D\_func | Diagnosis |
|-----------|---------|-----------|
| Low | Low | ✅ Ideal: consistent structure, stable optimisation |
| High | Low | ⚠ Acceptable: different structures, same ground state (symmetry) |
| Low | High | ❌ Landscape problem: structure consistent but optimisation unstable |
| High | High | ❌ Unreliable: random walk |

---

## Implemented Methods

### RL Methods

| Method | RL Algorithm | Action Space |
|--------|-------------|--------------|
| CRLQAS | DQN (off-policy) | Discrete |
| HyRLQAS / Hybrid\_REINFORCE | Batch REINFORCE (on-policy) | Hybrid discrete+continuous |
| RENEW | REINFORCE + Refine Head | Hybrid discrete+continuous |

### Non-RL Baselines

| Method | Type |
|--------|------|
| QuantumDARTS | Differentiable NAS (ICML 2023) |
| TFQAS | Training-free zero-cost proxy |

#### QuantumDARTS: nfev accounting

QuantumDARTS has two phases with fundamentally different evaluation semantics:

- **Phase 1 (Architecture Search):** Optimises soft Gumbel-softmax circuits via continuous relaxation.  These are *not* hardware-executable discrete circuits.  Phase 1 nfev is reported separately as `phase1_nfev` and is *not* directly comparable to RL method nfev.
- **Phase 2 (Discrete Evaluation):** Fixes architecture weights with argmax and evaluates real discrete circuits.  This is the only phase whose nfev is comparable to RL baselines and is reported as `phase2_nfev`.

Papers must report Phase 1 and Phase 2 nfev separately to avoid misleading comparisons.

---

## Installation

### Base environment

```bash
conda env create -f environment.yml
conda activate crlqas_env
```

### GPU qulacs (optional, recommended for n ≥ 10)

The PyPI build of `qulacs` is CPU-only.  To enable `QuantumStateGpu` on NVIDIA GPUs, build from source:

> **Tested:** NVIDIA L4, CUDA 12.8, Python 3.10

```bash
# 1. Install build dependencies
conda install -n crlqas_env -c conda-forge cuda-toolkit=12.8 boost boost-cpp -y
conda run -n crlqas_env pip install pybind11

# 2. Clone sources
git clone --depth=1 https://github.com/qulacs/qulacs.git /tmp/qulacs
git clone --depth=1 --branch v2.13.5 https://github.com/pybind/pybind11.git /tmp/pybind11_src

# 3. Apply two patches (upstream bugs)
sed -i 's/target_link_libraries(gpusim_static CUDA::cudart_static CUDA::curand_static CUDA::cublas_static)/target_link_libraries(gpusim_static CUDA::cudart_static CUDA::curand CUDA::cublas)/' \
    /tmp/qulacs/src/gpusim/CMakeLists.txt
sed -i 's|add_subdirectory(${pybind11_fetch_SOURCE_DIR})|add_subdirectory(${pybind11_fetch_SOURCE_DIR} ${CMAKE_BINARY_DIR}/_deps/pybind11_fetch-build)|' \
    /tmp/qulacs/CMakeLists.txt

# 4. Configure and build
mkdir -p /tmp/qulacs_build/_deps
cp -r /tmp/pybind11_src /tmp/qulacs_build/_deps/pybind11_fetch-src
conda run -n crlqas_env bash -c "
cd /tmp/qulacs_build
export CUDACXX=\$(which nvcc)
BOOST_DIR=\$(python -c 'import sysconfig; print(sysconfig.get_path(\"data\"))')
cmake /tmp/qulacs \
  -DPYTHON_EXECUTABLE=\$(which python) -DPYTHON_SETUP_FLAG=Yes -DUSE_GPU=Yes \
  -DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++ -DCMAKE_BUILD_TYPE=Release \
  -DBOOST_ROOT=\$BOOST_DIR -DFETCHCONTENT_FULLY_DISCONNECTED=ON
make -j\$(nproc) qulacs_core
SITE_PKG=\$(python -c 'import sysconfig; print(sysconfig.get_path(\"purelib\"))')
cp /tmp/qulacs_build/python/qulacs_core.cpython-310-x86_64-linux-gnu.so \$SITE_PKG/
cp -r /tmp/qulacs/pysrc/qulacs \$SITE_PKG/
"

# 5. Verify
python -c "from qulacs import QuantumStateGpu; print('GPU qulacs OK')"
```

`RLQAS/environment.py` automatically selects `QuantumStateGpu` when available; no code changes needed.  CPU-only installations fall back to `QuantumState` silently.

> If your Python version is not 3.10, rename the `.so` file accordingly (e.g. `cpython-311-...`).

---

## Quick Start

```bash
cd PSQASBench
conda activate crlqas_env

# CRLQAS on L1 H2, CPU
python main.py --method crlqas --mol L1_H2_Equil_4q --seed 11111 --device cpu

# CRLQAS on L6 BeH2 14q, GPU, parallel envs
python main.py --method crlqas --mol L6_BeH2_CCPVDZ_14q --seed 11111 --device cuda:0

# HyRLQAS (RENEW) on L2 LiH
python main.py --method hyrlqas --mol L2_LiH_Equil_6q --seed 11111 --device cuda:0

# Override config explicitly
python main.py --method crlqas --mol L6_BeH2_CCPVDZ_14q \
               --config bench_14q_rotosolve_gpu_k10.cfg --seed 11111 --device cuda:0

# Configs can live in subdirectories under configs/<method>/
python main.py --method crlqas --mol L1_BeH2_STO3G_6q \
               --config Depth_EXP/L1_BeH2_STO3G_6q_cobyla_depth10.cfg \
               --seed 11111 --device cuda:0 --use-wandb 0

# QuantumDARTS on L1 BeH2
python main.py --method qdarts --mol L1_BeH2_STO3G_6q --seed 11111 --device cuda:0
```

All output is written to:

```text
results/<method>/<mol>/<config_path_without_suffix>/seed<seed>/
```

Example:

```text
results/crlqas/L1_BeH2_STO3G_6q/Depth_EXP/L1_BeH2_STO3G_6q_cobyla_depth10/seed11111/
```

---

## Common Run Tweaks

### CLI flags

```bash
python main.py \
  --method crlqas \
  --mol L1_BeH2_STO3G_6q \
  --config Depth_EXP/L1_BeH2_STO3G_6q_cobyla_depth10.cfg \
  --seed 11111 \
  --device cuda:0 \
  --use-wandb 0 \
  --save-summary-detailed 0
```

Commonly changed flags:

- `--method`: choose benchmark runner (`crlqas`, `hyrlqas`, `qdarts`, `tfqas`)
- `--mol`: molecule key from `bench_utils.MOL_FILES`
- `--config`: config file relative to `configs/<method>/`
- `--seed`: random seed; creates a separate `seed<seed>` result directory
- `--device`: `cpu`, `cuda:0`, `cuda:1`, ...
- `--use-wandb 0`: disable Weights & Biases upload
- `--save-summary-detailed 1`: additionally save legacy `summary_<seed>.npy`

### Config fields most often edited

```ini
[general]
episodes = 10000
eval_every = 1000
eval_K = 50
num_parallel_envs = 8
use_wandb = 0
save_every = 500

[env]
num_layers = 20
accept_err = 0.0016
connectivity = linear    # required for L5 experiments only

[non_local_opt]
optim_alg = COBYLA
global_iters = 100
```

What they control:

- `general.episodes`: total training episodes
- `general.eval_every`: how often periodic eval runs
- `general.eval_K`: number of rollouts used per eval
- `general.num_parallel_envs`: parallel environments for training
- `env.num_layers`: maximum circuit depth (= maximum episode steps)
- `env.accept_err`: success threshold in Hartree
- `env.connectivity`: `all` (default) or `linear`; must be `linear` for L5 experiments
- `non_local_opt.optim_alg`: local angle optimizer (`COBYLA`, `Rotosolve`, `SPSA`, `AdamSPSA`, `PSRAdam`)
- `non_local_opt.global_iters`: optimizer budget for `COBYLA`, `SPSA`, `AdamSPSA`, `PSRAdam`
- `non_local_opt.rotosolve_sweeps`: sweep count for `Rotosolve`

---

## Configuration Reference

Config files live under `configs/crlqas/`, `configs/hyrlqas/`, `configs/qdarts/`, `configs/tfqas/`.

```ini
[general]
episodes = 10000          # total training episodes
eval_every = 1000         # periodic eval interval
eval_K = 20               # rollouts per eval
num_parallel_envs = 10    # 1 = single-env, >1 = parallel training
use_wandb = 1             # 0 = disable wandb upload
log_every = 10
save_every = 200

[env]
num_qubits = 14
num_layers = 20           # max circuit depth
accept_err = 0.0016       # chemical accuracy threshold (Ha)
connectivity = all        # all | linear

[problem]
mol_file = <filename>.npz

[agent]
agent_type = DeepQNstep
agent_class = DQN_Nstep
batch_size = 1000

[non_local_opt]
method = scipy_each_step
optim_alg = COBYLA        # COBYLA | Rotosolve | SPSA | AdamSPSA | PSRAdam
global_iters = 100
```

---

## Angle Optimisers

Each environment step runs a local angle optimiser over all current rotation gates.

| Optimiser | Update rule | Budget field | GPU acceleration |
|-----------|-------------|--------------|-----------------|
| `COBYLA` | SciPy derivative-free | `global_iters` | partial |
| `Rotosolve` | Analytic coordinate sweep using `{0, π/2, π}` probes | `rotosolve_sweeps` | strongest |
| `SPSA` | Stochastic gradient from `±delta` probes | `global_iters` | yes |
| `AdamSPSA` | SPSA gradient + Adam-style update | `global_iters` | yes |
| `PSRAdam` | Exact parameter-shift gradient + Adam | `global_iters` | yes |

When `num_parallel_envs > 1`, the runner uses one CUDA stream per environment and overlaps optimiser kernels before a single `synchronize()` barrier.  This grouped path is cleanest for `Rotosolve`, `SPSA`/`AdamSPSA`, and `PSRAdam`.

### Optimizer Field Reference

| Field | Used by | Meaning |
|-------|---------|---------|
| `method` | all | usually `scipy_each_step` |
| `optim_alg` | all | selects the local optimizer |
| `global_iters` | COBYLA, SPSA, AdamSPSA, PSRAdam | iteration budget |
| `rotosolve_sweeps` | Rotosolve | number of full coordinate sweeps |
| `global_batched_rotosolve` | Rotosolve, parallel | enable grouped batched path |
| `global_batched_spsa` | SPSA/AdamSPSA, parallel | enable grouped batched path |
| `global_batched_psr` | PSRAdam, parallel | enable grouped batched path |
| `a`, `alpha`, `c`, `gamma`, `lamda` | SPSA, AdamSPSA | SPSA schedule hyperparameters |
| `beta_1`, `beta_2` | AdamSPSA, PSRAdam | Adam momentum hyperparameters |
| `lr` | PSRAdam | Adam learning rate |

---

## Result Artifacts

Runs write their outputs under:

```text
results/<method>/<mol>/<config>/seed<seed>/
```

For RLQAS methods (`crlqas`, `hyrlqas`), the full training trace is written.  `TFQAS`
and `QuantumDARTS` also now write compatibility files for post-hoc structure analysis,
but their trace semantics are different: they serialize candidate/final circuits as
pseudo-episodes rather than logging an RL training trajectory.

Common files you will typically find are:

| File | Contents |
|------|----------|
| `run_meta.txt` | Method, mol, seed, device, exact energy, wall-clock time, final result |
| `episode_summary.tsv` | Per-episode energy, depth, CNOT count, reward, ε |
| `episode_traces.txt` | RL traces or compatibility pseudo-traces; may contain `analysis_snapshots` and/or `gates_direct` |
| `policy_loss.tsv` | Policy gradient / DQN loss per update |
| `best_train.txt` | Circuit achieving the lowest training energy |
| `best_eval.txt` | Best eval checkpoint (SR, CNOT@chem, D\_struct, D\_func) + full eval trend |
| `global_best_state_<seed>.npz` | Saved state tensor and op\_history of the best found circuit |
| `best_thresh*_model.pth` | Policy network checkpoint at global-best energy |
| `config_used.cfg` | Exact config file used (for reproducibility) |

### `episode_traces.txt` format

For RLQAS methods, each episode block may contain fields such as:

```
[episode N]
actions = [...]
energies_ha = [...]
energy_errors_ha = [...]
rewards = [...]
analysis_snapshots = [
  {"step": S, "gate_params": [...], "param_step_indices": [...]},
  ...
]
```

`analysis_snapshots` stores one or more threshold-crossing events.  Each snapshot records the
gate parameters needed for warm-start reconstruction in post-hoc analysis.

For `TFQAS` and `QuantumDARTS`, `episode_traces.txt` is written in a compatibility form:

```
[episode N]
energy_errors_ha = [...]
analysis_snapshots = [{"step": 0, "gates_direct": [...]}]
```

Here `gates_direct` is a direct gate list, so `critical_structure_tool` can reconstruct the
circuit without an RL action dictionary.

---

## Critical Structure Tool

### What it is

`critical_structure_tool` is a post-hoc analysis tool for answering:

> For a given performance regime, which gate sub-structure is actually responsible for the low energy?

It addresses the **puzzle-piece phenomenon** observed in RLQAS training traces: energy stays near the initial value for many steps, then drops sharply when one specific gate is inserted.  The tool identifies and compares these critical substructures across multiple episodes and seeds.

### Scope and prerequisites

The tool works with:

- **RLQAS methods** (`crlqas`, `hyrlqas`, `renew`) using action-based traces
- **direct-gate snapshot methods** (`TFQAS`, `QuantumDARTS`) that export compatibility traces with `gates_direct`

Required files per run directory:

| File | Required | Used for |
|------|---------|---------|
| `episode_traces.txt` | **yes** | RL action traces or direct-gate snapshot events |
| `config_used.cfg` | **yes** | action-id → gate decoding, molecule lookup, optimizer inheritance |
| `run_meta.txt` | optional | fallback `accept_err` / `analysis_save_threshold` when `--target-error-mha` is not specified |

For `TFQAS` and `QuantumDARTS`, the tool analyzes serialized candidate/final circuits rather than an RL trajectory.  These runs are therefore supported for structural pruning, but `episode` index should be interpreted as a method-specific candidate ordering rather than training time.

### Warm-start reconstruction

When `episode_traces.txt` contains parameterized `analysis_snapshots` (produced by runs after the snapshot-logging change), the tool uses the saved optimised angles as the starting point for circuit reconstruction.  This substantially improves reconstruction fidelity for near-degenerate systems (L3) where cold-start re-optimisation from angle=0 typically falls into a different basin.

For **old result files** without `analysis_snapshots`, the tool falls back to legacy `first_hit_snapshot` if present, and otherwise to cold-start reconstruction (all angles initialised to 0).

### Usage

Recommended wrapper script:

```bash
cd PSQASBench
conda activate crlqas_env

# Interactive: prints bucket summary and prompts for selection
python analyze_critical_structure.py results/crlqas/L1_BeH2_STO3G_6q/Depth_EXP/L1_BeH2_STO3G_6q_cobyla_depth10

# Direct bucket selection
python analyze_critical_structure.py \
  results/crlqas/L1_BeH2_STO3G_6q/Depth_EXP/L1_BeH2_STO3G_6q_cobyla_depth10 \
  --bucket 0.55 \
  --select-n 6 \
  --beam-width 4 \
  --branching-factor 3 \
  --prune-budget 1000 \
  --out-dir critical_structure_analysis/l1_beh2_cobyla_d10_bucket055

# Harder 8-qubit case — larger slack, smaller budget
python analyze_critical_structure.py \
  results/crlqas/L3_CH2_Singlet_8q/LevelCheck_EXP/L3_CH2_Singlet_8q_rotosolve_s2_check \
  --bucket 0.00 \
  --select-n 4 \
  --beam-width 4 \
  --branching-factor 3 \
  --prune-budget 300 \
  --reconstruction-slack-mha 0.5 \
  --out-dir critical_structure_analysis/l3_ch2_8q_bucket000

# Multiple run directories (multi-seed analysis)
python analyze_critical_structure.py \
  results/crlqas/L1_BeH2_STO3G_6q/Depth_EXP/L1_BeH2_STO3G_6q_cobyla_depth10 \
  results/crlqas/L1_BeH2_STO3G_6q/Depth_EXP/L1_BeH2_STO3G_6q_cobyla_depth10_seed2 \
  --bucket 0.55 --select-n 10 --out-dir critical_structure_analysis/l1_beh2_multiseed
```

Equivalent module entrypoint: `python -m critical_structure_tool <args...>`

### Key parameters

**Episode / bucket selection**

| Flag | Default | Meaning |
|------|---------|---------|
| `--target-error-mha` | inherit from `run_meta.txt` | threshold used to filter saved snapshot events |
| `--bucket` | interactive prompt | snapshot-event error bucket to analyse |
| `--select-n` | 6 | number of representative snapshot events to prune |
| `--late-fraction` | 1/3 | sample from the last fraction of episode indices; for TFQAS/QuantumDARTS this is candidate-order, not RL time |
| `--anchor-top-k` | 3 | most frequent event-time last actions treated as protected anchors |

**Error tolerances**

| Flag | Default | Meaning |
|------|---------|---------|
| `--bucket-slack-mha` | 0.05 | allowed error above bucket center during pruning |
| `--reconstruction-slack-mha` | 0.3 | extra slack when reconstructed baseline differs from trace; increase for L3+ |
| `--delta-tolerance-mha` | 0.2 | maximum single-step error increase for a gate to be deletable |

**Pruning budget**

| Flag | Default | Meaning |
|------|---------|---------|
| `--prune-budget` | 1000 | total child circuit evaluations across the whole pruning phase (main budget) |
| `--beam-width` | 4 | beam states kept after each expansion |
| `--branching-factor` | 3 | top-k deletion candidates expanded per beam state |
| `--max-prune-steps` | auto | depth cap per episode; auto-computed as `gate_count - fixed_gate_count` |

**Optimizer**

| Flag | Default | Meaning |
|------|---------|---------|
| `--analysis-optimizer` | `inherit` | optimizer used during reconstruction and pruning; `inherit` reads from `config_used.cfg` |
| `--cobyla-maxiter` | 300 | COBYLA iteration cap for analysis |
| `--rotosolve-sweeps` | 2 | Rotosolve sweep count for analysis |
| `--n-restarts` | 2 | COBYLA restarts per optimization call |

### Output files

All outputs are written under `--out-dir` (default: `critical_structure_analysis`):

| File | Contents |
|------|----------|
| `first_hit_error_distribution.tsv` | Fine-grained distribution of saved snapshot-event errors |
| `bucket_summary.tsv` | Coarser bucket view with counts and mean event step |
| `anchor_actions.tsv` | Most frequent event-time last actions for the chosen bucket |
| `selected_episodes.tsv` | Snapshot events selected for pruning (episode key, seed, step, last action) |
| `summary.tsv` | Per-snapshot pruning results: baseline error, retained error, gate counts, redundancy |
| `summary.md` | Human-readable markdown table of pruning results |
| `exact_signature_counts.tsv` | Exact retained gate-sequence matches across snapshot events |
| `meta.txt` | Full run metadata (parameters, anchor actions, common retained gates) |

### Interpreting results

**Redundancy ratio** — fraction of gates removed while preserving the error regime.  High redundancy (>70%) is the expected finding for RLQAS due to Circuit Structure Bias.

**Exact retained-structure matches** — episodes retaining identical gate sequences.  Count > 1 suggests a stable learned motif.

**Common retained gate signatures** — gates present in all pruned episodes (exact set intersection).  `none` does not mean no pattern exists; it often indicates consistent qubit-level patterns that the exact-match test misses (use the individual `summary.md` to inspect manually).

**Reconstruction baseline >> target error** — indicates warm-start failure (cold-start landed in wrong basin).  Increase `--reconstruction-slack-mha`, check that `analysis_snapshots` or legacy `first_hit_snapshot` is present in traces, or treat that episode as unusable.  This is expected for L3 near-degenerate molecules with old result files.

---

## Pipeline Overview

```text
Training run (CRLQAS / HyRLQAS / RENEW / TFQAS / QuantumDARTS)
    │
    ├── results/<method>/<mol>/<config>/seed<seed>/
    │   ├── episode_traces.txt     ← main input for analysis
    │   ├── config_used.cfg        ← gate decoding + optimizer info
    │   └── run_meta.txt           ← threshold fallback
    │
    └── critical_structure_tool
        │
        ├── 1. Discover runs (episode_traces.txt files)
        ├── 2. Expand saved snapshot events
        ├── 3. Build event-error distribution
        ├── 4. Choose bucket interactively or via --bucket
        ├── 5. Sample representative snapshot events
        ├── 6. Reconstruct circuit from actions or gates_direct
        │       warm-start: use analysis_snapshots angles when available
        │       legacy fallback: use first_hit_snapshot if present
        │       cold-start: re-optimize from angle=0 if no snapshot exists
        ├── 7. One-shot gate importance (delete each gate, measure |ΔE|)
        ├── 8. Beam search pruning (fixed deletion prior, prune-budget)
        └── 9. Compare retained structures across snapshot events
```

---

## Repository Structure

```text
PSQASBench/
├── main.py                        # unified entry point for all methods
├── bench_utils.py                 # molecule registry, arg parsing, path constants
│
├── RLQAS/                         # RL-based QAS runners (CRLQAS, HyRLQAS, RENEW)
│   ├── base_runner.py             # abstract BaseRunner + shared periodic_eval + analysis snapshot capture helpers
│   ├── crlqas_runner.py           # CRLQAS (DQN) runner
│   ├── hyrlqas_runner.py          # HyRLQAS / RENEW runner
│   ├── result_logger.py           # structured artifact writer (all methods share this)
│   ├── environment.py             # CircuitEnv (CRLQAS)
│   ├── hy_environment.py          # HyCircuitEnv (HyRLQAS, extends CircuitEnv)
│   └── agents/                    # DQN, HybridActionPolicy, HybridActionPolicywithRefine
│
├── QuantumDARTS/                  # Differentiable NAS baseline (ICML 2023)
│   ├── darts_runner.py            # QuantumDARTS runner (BaseRunner subclass)
│   ├── circuit.py                 # soft circuit with Gumbel-softmax mixing
│   └── gates.py                   # gate library
│
├── TFQAS/                         # Training-free QAS baseline
│   ├── tfqas_runner.py            # TFQAS runner (BaseRunner subclass)
│   ├── circuit.py                 # circuit template with DAG path counting
│   ├── expressibility.py          # expressibility proxy metric
│   ├── search_space.py            # search space definition
│   └── vqe_eval.py                # VQE evaluation helper
│
├── metrics/
│   ├── eval_utils.py              # greedy_rollout_k, stochastic_rollout_k, aggregate_metrics
│   └── pcd.py                     # D_struct / D_func via state-vector infidelity
│
├── critical_structure_tool/       # post-hoc critical circuit structure analysis
│   ├── cli.py                     # main entry point and argument parser
│   ├── circuit_utils.py           # gate construction, qulacs evaluation, optimizers
│   ├── pruning.py                 # gate importance + probabilistic beam pruning
│   ├── io_utils.py                # trace parsing, config reading, episode record construction
│   └── types.py                   # RunContext, SnapshotRecord, GateSpec, BranchState
│
├── analyze_critical_structure.py  # thin wrapper: python analyze_critical_structure.py <dirs...>
│
├── configs/
│   ├── crlqas/                    # .cfg files for CRLQAS experiments
│   ├── hyrlqas/                   # .cfg files for HyRLQAS experiments
│   ├── qdarts/                    # .cfg files for QuantumDARTS experiments
│   └── tfqas/                     # .cfg files for TFQAS experiments
│
├── mol_data/                      # pre-computed .npz Hamiltonians (Jordan-Wigner, 29 files)
├── mol_gen/                       # scripts to (re-)generate Hamiltonians and fingerprints
└── results/                       # training run outputs (gitignored data files)
```

---

## Known Systematic Issues (benchmark findings)

These are documented as findings; fixes are noted where planned:

1. **Circuit Structure Bias** — RL methods reach chemical accuracy but use far more gates than necessary.  Root cause: fixed max depth forces agents to use all steps; energy-only reward provides no incentive for circuit simplicity.  Diagnosed via `critical_structure_tool`; concept-proof fix is CRLQAS-STOP (planned).

2. **Checkpoint Selection Ambiguity** — All existing methods save checkpoints at global-best energy, biasing saved policies toward deep circuits.  No method implements Pareto-optimal checkpoint selection.  Documented as benchmark finding; not fixed.

3. **Training Instability** — Seed variance is large and largely unreported in prior work.  Quantified via SR@chem across ≥ 5 seeds per molecule × method.

4. **Curriculum Threshold Sensitivity** — The initial `accept_err` and tightening schedule affect convergence significantly but are rarely ablated.  Isolated via the LevelCheck experiment group in `configs/crlqas/LevelCheck_EXP/`.

5. **Reconstruction Fidelity (L3+)** — RLQAS circuits for near-degenerate molecules depend on specific angle trajectories accumulated during training.  Cold-start reconstruction in post-hoc analysis fails for many L3 episodes.  Partially addressed by the `analysis_snapshots` trace format (with legacy `first_hit_snapshot` fallback), which requires re-running experiments to populate for old runs.

