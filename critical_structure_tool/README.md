# Critical Structure Tool

This tool is used to analyze which circuit structure is most likely responsible for a successful RLQAS episode.

Instead of inspecting only one `best` circuit, it:

- scans one or more training-result directories,
- finds episodes that first enter a target error regime,
- samples representative episodes from late training,
- prunes those circuits with counterfactual delete-one-gate analysis,
- and checks whether different episodes retain the same core structure.

The main question it tries to answer is:

> For a given performance regime, what structure did the method repeatedly learn?

## What This Tool Uses

The tool currently reads:

- `episode_traces.txt`
- `config_used.cfg`
- `run_meta.txt` as an optional fallback

Role of each file:

- `episode_traces.txt`
  Main input. Used to read action sequences and per-step error traces.
- `config_used.cfg`
  Still needed to reconstruct action ids into gates and to inherit the training optimizer setup.
- `run_meta.txt`
  Only used as a fallback when `--target-error-mha` is not provided.

## Basic Usage

Recommended entrypoint:

```bash
conda run -n crlqas_env python analyze_critical_structure.py <result_dirs...>
```

Equivalent module entrypoint:

```bash
conda run -n crlqas_env python -m critical_structure_tool <result_dirs...>
```

### Example: Interactive Run

This prints the first-hit error distribution and bucket summary first, then lets you choose a bucket.

```bash
conda run -n crlqas_env python analyze_critical_structure.py \
  results/crlqas/L1_BeH2_STO3G_6q/Depth_EXP/L1_BeH2_STO3G_6q_cobyla_depth10 \
  --out-dir critical_structure_analysis/l1_beh2_interactive
```

### Example: Direct Analysis Of A Selected Bucket

```bash
conda run -n crlqas_env python analyze_critical_structure.py \
  results/crlqas/L1_BeH2_STO3G_6q/Depth_EXP/L1_BeH2_STO3G_6q_cobyla_depth10 \
  --bucket 0.55 \
  --select-n 6 \
  --beam-width 4 \
  --branching-factor 3 \
  --prune-budget 1000 \
  --out-dir critical_structure_analysis/l1_beh2_cobyla_d10_bucket055
```

### Example: Heavier 8-Qubit Case

```bash
conda run -n crlqas_env python analyze_critical_structure.py \
  results/crlqas/L3_CH2_Singlet_8q/LevelCheck_EXP/L3_CH2_Singlet_8q_rotosolve_s2_check \
  --bucket 0.00 \
  --select-n 4 \
  --beam-width 4 \
  --branching-factor 3 \
  --prune-budget 300 \
  --reconstruction-slack-mha 0.5 \
  --out-dir critical_structure_analysis/l3_ch2_8q_bucket000_main
```

## Key Hyperparameters

Only the most important knobs are listed here.

### Event and bucket selection

- `--target-error-mha`
  Defines the threshold used for the event.
  If omitted, the tool falls back to `accept_err` in `run_meta.txt`.

- `--bucket`
  Selects which first-hit error bucket to analyze.
  If omitted, the tool first prints the available buckets.

### Episode sampling

- `--select-n`
  Number of episodes to prune.

- `--late-fraction`
  Episodes are sampled from the last fraction of training.
  Current default is the last `1/3`.

### Anchor actions

- `--anchor-top-k`
  Number of most frequent hit-time last actions to treat as protected anchors.

### Error tolerance

- `--bucket-slack-mha`
  Additional tolerance relative to the chosen bucket center.

- `--reconstruction-slack-mha`
  Extra slack allowed because reconstructed baselines may differ from the trace-recorded first-hit error.
  This matters especially for harder systems.

- `--delta-tolerance-mha`
  Allowed single-step error degradation during pruning.

### Beam search pruning

- `--beam-width`
  Number of beam states kept after each expansion step.

- `--branching-factor`
  Here this should be interpreted as `top-k`.
  For each beam state, the tool expands using the top-k deletion candidates under the fixed deletion prior.
  The intended default design is:
  - `top-k = 3`

- `--prune-budget`
  Total pruning budget for one episode.
  This is the total number of child evaluations allowed during the whole beam-search pruning process.
  This is the main pruning budget.

- `--max-prune-steps`
  Optional override only.
  If omitted, the tool automatically uses:
  - `gate_count - fixed_gate_count`

### Analysis optimizer

- `--analysis-optimizer`
  Current choices:
  - `inherit`
  - `cobyla`
  - `rotosolve`

  Recommended default:
  - `inherit`

  That means the tool tries to use the same optimizer family as the original training run.

- `--cobyla-maxiter`
  Analysis-time COBYLA iteration cap.

- `--rotosolve-sweeps`
  Analysis-time Rotosolve sweep count.

## Main Outputs

Typical outputs are written under `--out-dir`.

- `first_hit_error_distribution.tsv`
  Sorted first-hit error distribution for the discovered runs.

- `bucket_summary.tsv`
  Coarser bucket summary for quick inspection.

- `anchor_actions.tsv`
  Most frequent hit-time last actions for the chosen bucket.

- `selected_episodes.tsv`
  The episodes selected for pruning.

- `meta.txt`
  Global metadata for this analysis run.

- `summary.tsv`
  Compact per-episode pruning results.

- `summary.md`
  Human-readable version of the pruning summary.

- `exact_signature_counts.tsv`
  Counts of exact retained structures after pruning.

## Implementation Idea

The current pipeline is:

### 1. Discover runs

The tool recursively finds all `episode_traces.txt` files under the provided paths.
Each parent directory is treated as one run.

### 2. Read the minimum required context

For each run:

- read traces from `episode_traces.txt`,
- read gate decoding and optimizer info from `config_used.cfg`,
- optionally read `accept_err` from `run_meta.txt` if no explicit target threshold was given.

### 3. Define the event

The current event is:

- the first step where `energy_error <= target_error_threshold`

This is not a full jump detector.
It is a practical proxy for the jump, because in many QAS traces the error drops suddenly instead of decreasing smoothly.

### 4. Build the first-hit error distribution

For every successful episode, the tool records:

- the first-hit step,
- the first-hit error,
- the action prefix up to that point,
- the last action at the hit time.

It then writes `first_hit_error_distribution.tsv` so the user can inspect the error landscape before choosing what regime to analyze.

### 5. Choose the target regime

The current implementation still analyzes one bucket at a time.
The bucket is based on first-hit error in mHa.

Conceptually, this bucket is meant to represent:

- a dominant learned mode,
- or a particular success regime that the user wants to inspect.

### 6. Sample representative episodes

Episodes are sampled from the last `late_fraction` part of training.
Within that late pool, the tool samples rather than always taking the final few episodes.

This is meant to bias the analysis toward learned late-stage behavior while avoiding overfitting to only the very last episodes.

### 7. Reconstruct the first-hit circuit

The tool reconstructs the circuit from the first-hit action prefix.
At this stage, only the discrete structure is faithfully recovered.
Continuous parameters are re-optimized later.

### 8. Re-optimize the reconstructed circuit

The reconstructed circuit is optimized again using the analysis optimizer.
The intended default behavior is to inherit the optimizer family and key settings from the training run.

This gives a reconstructed baseline circuit and a reconstructed baseline error.

### 9. One-shot counterfactual gate analysis

For the reconstructed baseline circuit, the tool removes each gate once, re-optimizes, and measures the error change.

This produces a one-shot importance estimate for every gate.

The current intended use of this step is:

- not to fully solve the structure problem,
- but to build a deletion prior for pruning.

### 10. Convert gate importance into a fixed deletion prior

The absolute counterfactual error change `|Δerror|` is treated as a measure of how risky it is to delete a gate.

Smaller `|Δerror|` means:

- the gate appears less important,
- so deleting it should be more likely.

This one-shot importance map is computed only once, to keep the total budget manageable.

### 11. Standard beam search pruning

Pruning is then performed as a standard beam search over deletion actions.

For each beam state:

- rank deletable gates using the fixed deletion prior,
- take the top-k candidates,
- delete one gate to form each child state,
- re-optimize the child circuit,
- reject the child if the retained error falls outside the tolerance window.

After expansion:

- keep only the top `beam-width` states,
- continue until the pruning budget is exhausted or no more valid deletions remain.

Important detail:

- `--branching-factor` is being used as `top-k`
- `--prune-budget` is the total evaluation budget
- `--max-prune-steps` should not be the main user-controlled budget

### 12. Compare retained structures across episodes

After pruning all selected episodes, the tool compares the retained structures:

- exact retained structure matches,
- common gate signatures,
- and whether there is a stable shared motif.

This is the final output used to judge whether a success regime corresponds to a consistent critical structure.

## Current Problem That Still Needs To Be Solved

The biggest unresolved problem is **reconstruction fidelity on harder systems**.

For easy systems such as `L1_BeH2_STO3G_6q`, the reconstructed baseline can still remain close to the trace-recorded first-hit regime, so the retained structures are often interpretable.

For harder cases such as `L3_CH2_Singlet_8q`, this often breaks:

- the trace may record a first-hit error in the `0.00 mHa` bucket,
- but after reconstructing the same action prefix and re-optimizing it from scratch, the tool may land in a completely different basin,
- producing a very large reconstructed baseline error.

This happens because the tool currently reconstructs:

- the discrete action prefix,
- but **not** the original warm-start parameter trajectory used during training.

So for difficult or branch-sensitive problems, the current tool may be pruning a circuit in the wrong basin.

That means:

- the current output can still be useful as a clue for branch diversity,
- but it is not yet strong enough to support high-confidence causal claims about the true critical structure of the original training trajectory.

This problem needs to be solved.

The most likely long-term fix is:

- saving more faithful intermediate circuit/parameter snapshots during training,
- so the analysis tool can reconstruct not only the discrete circuit structure, but also a basin-faithful starting point for counterfactual pruning.
