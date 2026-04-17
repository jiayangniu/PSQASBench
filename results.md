# Level 1 Results

`L1_BeH2_STO3G_6q` depth-scan summary extracted from each run folder's `best_train.txt`.

| Method | Optimizer | Config | MaxStep | BestDepth | Error (Ha) | Error (mHa) |
|---|---|---|---:|---:|---:|---:|
| crlqas | COBYLA | L1_BeH2_STO3G_6q_cobyla_depth10 | 10 | 9 | 0.000268225 | 0.268 |
| crlqas | COBYLA | L1_BeH2_STO3G_6q_cobyla_depth20 | 20 | 15 | 0.000268226 | 0.268 |
| crlqas | COBYLA | L1_BeH2_STO3G_6q_cobyla_depth30 | 30 | 11 | 0.000268225 | 0.268 |
| crlqas | COBYLA | L1_BeH2_STO3G_6q_cobyla_depth40 | 40 | 20 | 0.000268225 | 0.268 |
| crlqas | COBYLA | L1_BeH2_STO3G_6q_cobyla_depth50 | 50 | 27 | 0.000132432 | 0.132 |
| crlqas | Rotosolve | L1_BeH2_STO3G_6q_rotosolve_depth10 | 10 | 9 | 0.000554443 | 0.554 |
| crlqas | Rotosolve | L1_BeH2_STO3G_6q_rotosolve_depth20 | 20 | 11 | 0.000554442 | 0.554 |
| crlqas | Rotosolve | L1_BeH2_STO3G_6q_rotosolve_depth30 | 30 | 17 | 0.000554395 | 0.554 |
| crlqas | Rotosolve | L1_BeH2_STO3G_6q_rotosolve_depth40 | 40 | 30 | 0.000550595 | 0.551 |
| crlqas | Rotosolve | L1_BeH2_STO3G_6q_rotosolve_depth50 | 50 | 15 | 0.000270796 | 0.271 |
| hyrlqas | COBYLA | L1_BeH2_STO3G_6q_cobyla_depth10 | 10 | 6 | 0.000554200 | 0.554 |
| hyrlqas | COBYLA | L1_BeH2_STO3G_6q_cobyla_depth20 | 20 | 19 | 0.000288125 | 0.288 |
| hyrlqas | COBYLA | L1_BeH2_STO3G_6q_cobyla_depth30 | 30 | 21 | 0.000275727 | 0.276 |
| hyrlqas | COBYLA | L1_BeH2_STO3G_6q_cobyla_depth40 | 40 | 14 | 0.000270959 | 0.271 |
| hyrlqas | COBYLA | L1_BeH2_STO3G_6q_cobyla_depth50 | 50 | 23 | 0.000273820 | 0.274 |
| hyrlqas | Rotosolve | L1_BeH2_STO3G_6q_rotosolve_depth10 | 10 | 9 | 0.000554200 | 0.554 |
| hyrlqas | Rotosolve | L1_BeH2_STO3G_6q_rotosolve_depth20 | 20 | 18 | 0.000436898 | 0.437 |
| hyrlqas | Rotosolve | L1_BeH2_STO3G_6q_rotosolve_depth30 | 30 | 19 | 0.000270005 | 0.270 |
| hyrlqas | Rotosolve | L1_BeH2_STO3G_6q_rotosolve_depth40 | 40 | 16 | 0.000270005 | 0.270 |
| hyrlqas | Rotosolve | L1_BeH2_STO3G_6q_rotosolve_depth50 | 50 | 14 | 0.000268098 | 0.268 |

## Level 1 Rotosolve Sweep=3 Results

The following runs use the same `L1_BeH2_STO3G_6q` depth-scan setup as above,
but increase `rotosolve_sweeps` from `1` to `3`.

| Method | Optimizer | Config | MaxStep | BestDepth | Error (Ha) | Error (mHa) |
|---|---|---|---:|---:|---:|---:|
| crlqas | Rotosolve(s3) | L1_BeH2_STO3G_6q_rotosolve_s3_depth10 | 10 | 8 | 0.000554443 | 0.554 |
| crlqas | Rotosolve(s3) | L1_BeH2_STO3G_6q_rotosolve_s3_depth20 | 20 | 11 | 0.000554443 | 0.554 |
| crlqas | Rotosolve(s3) | L1_BeH2_STO3G_6q_rotosolve_s3_depth30 | 30 | 20 | 0.000554443 | 0.554 |
| crlqas | Rotosolve(s3) | L1_BeH2_STO3G_6q_rotosolve_s3_depth40 | 40 | 25 | 0.000554395 | 0.554 |
| crlqas | Rotosolve(s3) | L1_BeH2_STO3G_6q_rotosolve_s3_depth50 | 50 | 17 | 0.000554246 | 0.554 |
| hyrlqas | Rotosolve(s3) | L1_BeH2_STO3G_6q_rotosolve_s3_depth10 | 10 | 8 | 0.000554200 | 0.554 |
| hyrlqas | Rotosolve(s3) | L1_BeH2_STO3G_6q_rotosolve_s3_depth20 | 20 | 13 | 0.000268098 | 0.268 |
| hyrlqas | Rotosolve(s3) | L1_BeH2_STO3G_6q_rotosolve_s3_depth30 | 30 | 16 | 0.000268098 | 0.268 |
| hyrlqas | Rotosolve(s3) | L1_BeH2_STO3G_6q_rotosolve_s3_depth40 | 40 | 31 | 0.000268098 | 0.268 |
| hyrlqas | Rotosolve(s3) | L1_BeH2_STO3G_6q_rotosolve_s3_depth50 | 50 | 28 | 0.000268098 | 0.268 |

## Level 1 Chemical-Accuracy Trigger Actions

The following summary aggregates the original 20 completed `L1_BeH2_STO3G_6q` depth-scan runs (`crlqas/hyrlqas × COBYLA/Rotosolve × depth10/20/30/40/50`) and counts, for each episode, which action first reaches chemical accuracy.

### Global Top Actions

| Rank | Count | Action ID | Decoded Action |
|---|---:|---:|---|
| 1 | 45291 | 46 | `RY(q=5)` |
| 2 | 41434 | 43 | `RY(q=4)` |
| 3 | 36972 | 42 | `RX(q=4)` |
| 4 | 22617 | 20 | `CNOT(ctrl=4, targ=5, offset=1)` |
| 5 | 19032 | 45 | `RX(q=5)` |
| 6 | 14890 | 29 | `CNOT(ctrl=5, targ=4, offset=5)` |

These top-6 actions account for `180236 / 189236 ≈ 95.2%` of all first-hit
chemical-accuracy events.

### Grouped Pattern Summary

| Group | Total Hit Episodes | Dominant Trigger Actions |
|---|---:|---|
| `crlqas / COBYLA` | 48442 | `RY(q=5)`, `RX(q=5)`, `CNOT(4->5)`, `RX(q=4)`, `RY(q=4)` |
| `crlqas / Rotosolve` | 47238 | `CNOT(5->4)`, `RY(q=4)`, `RY(q=5)`, `RX(q=4)`, `CNOT(4->5)` |
| `hyrlqas / COBYLA` | 48409 | `RY(q=4)`, `RY(q=5)`, `RX(q=4)` strongly dominate |
| `hyrlqas / Rotosolve` | 45147 | `RX(q=4)`, `RY(q=5)`, `CNOT(4->5)`, `RY(q=4)` dominate |

### Takeaway

The Level 1 BeH2 task reaches chemical accuracy through a highly concentrated local motif rather than through diverse circuit endings. Across methods, optimizers, and depth budgets, the trigger actions overwhelmingly focus on:

- single-qubit rotations on `q=4` and `q=5`
- two-qubit entanglers between `q=4` and `q=5`

This suggests that `L1_BeH2_STO3G_6q` behaves more like a fixed-pattern entry benchmark than a high-diversity structure-search challenge.

## Level 1 Interpretation: Shallow Depth May Be Better

For `L1_BeH2_STO3G_6q`, the current results suggest that success is often tied to discovering a specific shallow motif rather than exploiting a genuinely deep expressive circuit family.

Under this interpretation, increasing the allowed depth has two opposite effects:

- it creates more opportunities to stumble upon the key structure;
- it also enlarges the search space and increases interference from already
  inserted gates, making later exploration and credit assignment harder.

This means deeper search does not necessarily help Level 1. If the task is mostly about discovering one compact structure, then a shallow depth budget with more episodes may be more appropriate than a large depth budget with the same training horizon.

### Working Hypothesis

For Level 1 BeH2, a protocol with:

- shallow fixed `max depth`
- larger episode budget

may better reflect the real difficulty of the task than a protocol that mainly increases depth.

### What This Would Mean for Level 1

If this hypothesis holds, then Level 1 should primarily test:

- whether a method can efficiently discover the key shallow motif;
- whether it can do so stably across many episodes and seeds;
- whether it reaches chemical accuracy without relying on unnecessarily deep
  circuits.

In that case, Level 1 is better viewed as a **shallow-structure discovery and sample-efficiency benchmark**, rather than a deep-circuit expressivity test.

### Suggested Validation Direction

The next validation step should compare:

- shallow depth budgets such as `6 / 8 / 10`
- larger episode budgets

against the current deeper-budget setting, using metrics such as:

- success rate within budget;
- episodes to first chemical accuracy;
- time to first chemical accuracy;
- depth at first chemical-accuracy hit.

## Level 1 Benchmarking Note: Learned Modes vs Incidental Modes

The first-hit error bucket analysis suggests that, for benchmarking purposes, we should not treat all chemical-accuracy buckets equally.

### Core Distinction

We should separate:

- **dominant learned modes**: high-support buckets that appear across many episodes and therefore reflect a structure that RLQAS repeatedly learns;
- **incidental modes**: low-support buckets that appear only once or a few times and are therefore more likely to reflect accidental discovery than a stable learned policy.

Under this interpretation, buckets with only `1-2` or a few supporting episodes should not be used as the main evidence for model capability in the benchmark. They are still interesting, but they more likely indicate that the agent *happened to hit* a good structure rather than *learned how to produce* that structure reliably.

### What Matters Most for Benchmarking

For the Level 1 benchmark, the most informative buckets are the high-frequency ones, because these reveal the structure families that RLQAS can learn consistently. These buckets are much more representative of the real search behavior of the method.

In the current **first-hit** view, the most important dominant mode is the `~0.55 mHa` bucket, because it has overwhelming support and therefore reflects the main learned chemical-accuracy pattern. By contrast, much smaller buckets such as `0.27 mHa` or `0.13 mHa` are promising but do **not** yet constitute evidence of a stable learned mode under the first-hit criterion alone.

### Implication

This means that the benchmark discussion should explicitly state:

- high-support buckets are the primary evidence for RLQAS capability;
- very low-support buckets should be treated as exploratory or accidental discoveries;
- if we want to analyze the better but rarer `0.27 / 0.13 mHa` solutions as learned high-quality paradigms, we need a second analysis based on **best-quality / minimum-error** outcomes rather than first-hit outcomes.

So, for now, the safer benchmarking interpretation is:

- `0.55 mHa` reflects the dominant learned first-hit paradigm;
- rarer lower-error paradigms remain interesting, but they should be presented as candidate higher-quality motifs that require further validation rather than as the main learned behavior of the method.

## Level 1 Interim Positioning Notes

The current evidence is strong enough to **fix `BeH2` as Level 1**, but the precise wording of the Level 1 positioning should remain provisional until the usability of all levels has been checked.

### What Already Seems Clear

- `L1_BeH2_STO3G_6q` is a **qualified entry molecule**: mainstream methods can cross chemical accuracy on it, so it functions as a valid access-level task rather than an artificially impossible one.
- The most interesting differences on BeH2 emerge **after** methods cross chemical accuracy, not before.
- This makes Level 1 less suitable as a pure "who has the lowest final error" benchmark and more suitable as a task for studying what kind of circuit family the method actually learns.

### Why Level 1 Is Special

For BeH2, the key questions are not only solvability, but also:

- whether the learned structure is interpretable;
- whether the same high-level motif is learned repeatedly;
- whether reported strong performance reflects a stable learned paradigm or a small number of lucky episodes;
- whether RL-based QAS evaluation should emphasize dominant learned modes rather than accidental best-case outcomes.

In other words, Level 1 naturally opens up several discussion directions:

- **entry-level solvability**
- **motif-level interpretability**
- **structure stability / reproducibility**
- **evaluation bias caused by stochastic best-case reporting**

### Temporary Working View

Until the remaining levels are fully checked, the safest temporary view is:

- keep `BeH2` as **Level 1**;
- treat Level 1 as an **entry-level, interpretable, post-chemical-accuracy analysis task**;
- avoid over-fixing the final paper wording until we see whether the other levels support a broader **hierarchical diagnostic benchmark** narrative.

### What To Revisit Later

After all levels are validated, we should revisit:

- the final one-sentence definition of Level 1;
- which Level 1 metrics are primary versus secondary;
- whether Level 1 should explicitly be framed as an evaluation-calibration level for RLQAS;
- how much of the BeH2 interpretability story belongs in the main paper versus supplementary analysis.

## Full-Space Fingerprints

This section intentionally keeps only the unrestricted **Full-Space** fingerprint
metrics. Physical-sector indicators are omitted here by design.

Columns:

- `Gap01`: `E1 - E0`
- `Hub`: qubit hub score
- `Asym`: qubit asymmetry
- `Z-only / XY-only / Mixed`: Pauli-weight fractions
- `>=4-body`: Pauli-weight fraction carried by terms with at least 4 Pauli factors
- `G1-G4`: Gershgorin-style matrix diagnostics

### Current Benchmark Suite

| Molecule | q | Gap01 | Hub | Asym | Z-only | XY-only | Mixed | >=4-body | G1 | G2 | G3 | G4 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `L1_BeH2_STO3G_6q` | 6 | 0.2120 | 1.1187 | 0.3560 | 0.9784 | 0.0216 | 0.0000 | 0.0216 | 0.9688 | 0.0937 | 29.5711 | 0.1676 |
| `L2_LiH_Equil_6q` | 6 | 0.0772 | 1.1349 | 0.2556 | 0.8759 | 0.0648 | 0.0594 | 0.0847 | 0.8906 | 0.2249 | 16.3985 | -0.0083 |
| `L3_CH2_Singlet_6q` | 6 | 0.0000 | 1.1277 | 0.2113 | 0.9615 | 0.0385 | 0.0000 | 0.0385 | 0.9688 | 0.6335 | inf | -0.0944 |
| `L4_H2_Stretch_4q` | 4 | 0.0044 | 1.0186 | 0.0371 | 0.7280 | 0.2720 | 0.0000 | 0.2720 | 0.7500 | 0.1800 | inf | -0.0535 |
| `L4_H2O_StrongCorr_8q` | 8 | 0.0939 | 1.3888 | 0.6568 | 0.7235 | 0.1011 | 0.1754 | 0.2168 | 0.6484 | 0.0738 | 3.4383 | -0.5969 |
| `L5_H4_Chain_8q` | 8 | 0.2326 | 1.1188 | 0.3247 | 0.5989 | 0.1487 | 0.2524 | 0.4011 | 0.4180 | 0.0109 | 2.1014 | -1.7976 |
| `L6_BeH2_631G_8q` | 8 | 0.0895 | 1.1145 | 0.2283 | 0.9530 | 0.0470 | 0.0000 | 0.0470 | 0.9805 | 0.0462 | 16.3786 | -0.0270 |
| `L6_BeH2_6311G_10q` | 10 | 0.0627 | 1.0956 | 0.2531 | 0.9231 | 0.0349 | 0.0420 | 0.0769 | 0.9629 | 0.0105 | 26.5974 | -0.0691 |
| `L6_BeH2_CCPVDZ_12q` | 12 | 0.0677 | 1.1430 | 0.2538 | 0.8889 | 0.0303 | 0.0808 | 0.1111 | 0.9043 | 0.0192 | 22.3112 | -0.1791 |
| `L6_BeH2_CCPVDZ_14q` | 14 | 0.0615 | 1.1333 | 0.3765 | 0.7671 | 0.0520 | 0.1808 | 0.2329 | N/A | N/A | N/A | N/A |

`L6_BeH2_CCPVDZ_14q` keeps its Pauli-decomposition metrics here, but `G1-G4`
are left `N/A` in this lightweight summary pass because the dense `14q`
Hamiltonian is too heavy for routine table generation.

### Active Candidate / Replacement Pool

These are the molecules that are currently being actively discussed as Level-3
or Level-4 alternatives.

| Molecule | q | Gap01 | Hub | Asym | Z-only | XY-only | Mixed | >=4-body | G1 | G2 | G3 | G4 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `L3_CH2_Singlet_R113_A080_6q` | 6 | 0.0107 | 1.1162 | 0.2021 | 0.9586 | 0.0414 | 0.0000 | 0.0414 | 0.9844 | 0.4997 | inf | -0.1319 |
| `L3_CH2_Singlet_R113_A130_6q` | 6 | 0.0000 | 1.1665 | 0.2450 | 0.8998 | 0.0468 | 0.0534 | 0.0802 | 0.9375 | 0.2339 | 18.6190 | -0.1987 |
| `L3_CH2_Singlet_R130_A100_6q` | 6 | 0.0000 | 1.1081 | 0.1959 | 0.9572 | 0.0428 | 0.0000 | 0.0428 | 0.9688 | 0.0777 | inf | -0.0988 |
| `L3_CH2_Singlet_R130_A130_6q` | 6 | 0.0000 | 1.1686 | 0.2554 | 0.8747 | 0.0532 | 0.0722 | 0.0967 | 0.9062 | 0.1852 | 11.5277 | -0.2878 |
| `L3_CH2_Singlet_8q` | 8 | 0.0000 | 1.1616 | 0.3269 | 0.9067 | 0.0435 | 0.0499 | 0.0933 | 0.9453 | 0.2099 | 39.9195 | -0.3277 |
| `L3_LiH_Stretch_6q` | 6 | 0.0000 | 1.2859 | 0.4704 | 0.7079 | 0.1236 | 0.1684 | 0.1756 | 0.5938 | 0.0774 | inf | -0.1277 |
| `L3_H3_Triangle_6q` | 6 | 0.0000 | 1.1436 | 0.2333 | 0.7895 | 0.1296 | 0.0809 | 0.2105 | 0.8125 | 0.4447 | 2.9728 | -0.6707 |
| `L4_H3_Linear_6q` | 6 | 0.0000 | 1.1462 | 0.2298 | 0.7091 | 0.1544 | 0.1366 | 0.2909 | 0.6562 | 0.0209 | 2.8517 | -0.8334 |

### CH2 6q -> 8q Expansion Note

To test whether the current `CH2` candidate is too easy mainly because of the
small `CAS(2e,3o)` truncation, we also generated an `8q` version by enlarging
the active space to `CAS(2e,4o)` while keeping the original geometry fixed.

Compared with `L3_CH2_Singlet_6q`, the new `L3_CH2_Singlet_8q` shifts in the
expected "harder" direction:

- `Z-only`: `0.9615 -> 0.9067`
- `Mixed`: `0.0000 -> 0.0499`
- `>=4-body`: `0.0385 -> 0.0933`
- `Asym`: `0.2113 -> 0.3269`
- `G2`: `0.6335 -> 0.2099`

So even before running the new training sweep, the fingerprint view suggests
that the `8q` active-space expansion is a more meaningful difficulty test than
the original `6q` CH2 baseline.

### Additional Registry / Legacy Molecules

These are still part of the project registry and are useful as auxiliary
references, but they are not the current main benchmark choices.

| Molecule | q | Gap01 | Hub | Asym | Z-only | XY-only | Mixed | >=4-body | G1 | G2 | G3 | G4 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `L1_H2_Equil_4q` | 4 | 0.6009 | 1.0353 | 0.0707 | 0.9045 | 0.0955 | 0.0000 | 0.0955 | 1.0000 | 1.3989 | 5.6730 | 0.3997 |
| `L1_BH_6q` | 6 | 0.0349 | 1.0292 | 0.0877 | 0.9565 | 0.0435 | 0.0000 | 0.0435 | 1.0000 | 2.3754 | 6.1132 | -0.1140 |
| `L2_BeH_Plus_4q` | 4 | 0.0000 | 1.1634 | 0.3545 | 0.8803 | 0.0431 | 0.0765 | 0.0431 | 1.0000 | 1.4930 | 25.9441 | -0.0535 |
| `L2_BF_8q` | 8 | 0.1153 | 1.1896 | 0.4449 | 0.8676 | 0.0244 | 0.1080 | 0.1324 | 0.8984 | 0.3400 | 15.3662 | -0.1064 |
| `L3_HeH_Plus_4q` | 4 | 0.0000 | 1.2260 | 0.4990 | 0.8373 | 0.0720 | 0.0907 | 0.0720 | 0.8750 | 0.9875 | 39.4561 | -0.0745 |
| `L5_H3_Linear_6q` | 6 | 0.0000 | 1.1462 | 0.2298 | 0.7091 | 0.1544 | 0.1366 | 0.2909 | 0.6562 | 0.0209 | 2.8517 | -0.8334 |
| `L6_BeH2_10q` | 10 | 0.2060 | 1.2149 | 0.4362 | 0.8893 | 0.0410 | 0.0698 | 0.1107 | 0.8730 | 0.0200 | 18.2063 | -0.2819 |

## Proposed Critical-Structure Analysis Direction

The current version of `analyze_critical_structure.py` is useful for studying a
single best circuit, but it does **not** yet answer the benchmark question we
now care about most:

- not "what is the critical gate in one lucky best-case circuit?"
- but "what is the critical structure that the method repeatedly learns when it
  successfully crosses a major energy barrier?"

### Working Hypothesis

For Level-1 / Level-3 style analysis, the most meaningful critical structure is
unlikely to come from the single best-energy episode. The better target is the
**dominant post-jump success mode**:

- first detect large error drops ("jumps") inside episodes;
- record the `last action` at the jump;
- treat that `last action` as a likely component of the important structure;
- bucket episodes by the error immediately after the jump;
- select the **most populated post-jump bucket** rather than the single best
  episode;
- draw a batch of representative circuits from that bucket and analyse them
  together.

The reason for this choice is that the single best circuit may be accidental,
while the dominant bucket is more likely to reflect what the method actually
learns in a stable and repeatable way.

### Proposed Pipeline

1. Parse all episode traces and locate the largest jump in each episode.
2. Collect the jump-time `last action`, `last2`, `last3`, and jump-following
   error.
3. Bucket jump-following errors and identify the dominant success bucket.
4. Within that dominant bucket, count which `last actions` are most frequent.
   These actions should be marked as **anchor / important actions**.
5. Select a batch of representative circuits from that bucket instead of using
   only the global best circuit.
6. Run counterfactual gate-removal analysis on those circuits, but initially
   try to avoid removing the high-frequency anchor actions until later rounds.
7. Compare the structures that remain after deletion across the whole batch.
   If the retained core motifs are consistent, that shared motif is a much
   better candidate for the true critical structure.

### Why This Matters

This approach is better aligned with the benchmarking goal:

- it separates **stable learned structure** from **accidental best-case
  structure**;
- it uses jump statistics to locate likely key actions;
- it lets us validate whether the same minimal motif survives across many
  successful circuits;
- it is especially relevant for near-degenerate problems such as `CH2`, where
  the main question is not simply how low the final error becomes, but what
  structural obstacle the method had to overcome.

### Current Conclusion

The next version of the critical-structure tool should move away from
single-circuit best-case ablation and toward:

- **jump-driven structure mining**
- **dominant-bucket circuit selection**
- **protected / anchor-aware counterfactual gate removal**
- **cross-episode consistency checking of the surviving motif**

This is now the preferred analysis direction for interpreting Level-1 and
Level-3 benchmark behavior.
