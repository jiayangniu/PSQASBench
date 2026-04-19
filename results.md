# Benchmark Results

## Fixed Level Choices

The current benchmark choices that are no longer under active debate are:

- `Level 1 = L1_BeH2_STO3G_6q`
- `Level 3 = L3_CH2_Singlet_8q`
- `Level 4 = L4_H2O_StrongCorr_8q`
- `Level 6 = BeH2 scalability gradient (8q → 10q → 12q → 14q)`

These choices are reflected below in both the fingerprint tables and the critical-structure notes.

## Level 1 Reference Results

The following `Level 1` table is kept as a compact historical reference for the original `BeH2` depth-scan runs. The benchmark choice itself is already fixed to `L1_BeH2_STO3G_6q`, so this section is retained for lookup rather than as an active discussion section.

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

## Critical-Structure Findings

The archived note in [`critical_structure_analysis_old/findings.md`](/home/ubuntu/NeurIPS2026/PSQASBench/critical_structure_analysis_old/findings.md:1) is now summarized here as the current interpretation of the tool outputs.

### L1: BeH2 is a concentrated local-motif benchmark

For `L1_BeH2_STO3G_6q`, the critical-structure analysis continues to support a very concentrated success family:

- the dominant successful structures stay on qubits `{4, 5}`;
- the key operations are rotations on `q=4/q=5` and entanglers on the `4↔5`  edge;
- the task behaves more like a compact local-motif discovery problem than a diverse circuit-search benchmark.

This is why the older exploratory depth-scan tables and trigger-action dumps are no longer the main story here: the benchmark choice itself is already fixed, and the more important conclusion is that Level 1 is structurally concentrated.

### L2: LiH shows a near-threshold q4/q5 bias, but better solutions need q2

For `L2_LiH_Equil_6q`, the recent curriculum run and the updated tool outputs suggest a useful mismatch between "what RLQAS likes to place on average" and "what lower-error retained structures actually require."

The late-training gate-bias statistics over the last training third show a clear placement preference on `q4/q5`:

- the most common rotation targets are `q4` and `q5`;
- the most frequent retained edge motifs in the late average are `CNOT(4->5)` and `CNOT(5->4)`;
- `q4/q5` are touched in essentially every late-training episode.

This is directionally consistent with the LiH asymmetry descriptors in the fingerprint table: the molecule is not fully symmetric (`Asym = 0.2556`), and RLQAS does not distribute gates uniformly across all qubits.

However, the retained-structure comparison across error regimes shows that the "easy-to-find" near-threshold solution family is not the same as the better low-error family:

- in the `1.06 mHa` bucket, the pruned structures are usually very small and are dominated by `q4/q5` operations such as `RX(q=4)`, `RX(q=5)`, `RY(q=4)`, `RY(q=5)`, and `CNOT(4->5)`;
- in the `0.1-0.5 mHa` regime, the retained structures become much more diverse and repeatedly bring in `q2` through operations such as `RX(q=2)`, `0->2`, `1->2`, `2->0`, `2->3`, `2->4`, `2->5`, and `5->2`;
- this matters because the molecular fingerprint already marks `q2` as one of the most important qubits in LiH, while the average late-training gate-placement statistics are still more heavily concentrated on `q4/q5`.

So the current LiH interpretation is not that RLQAS is "wrong," but that it shows a plausible **search bias**:

- the method appears to find a relatively cheap `q4/q5`-centered near-threshold family first;
- lower-error structures still require the more globally relevant `q2` interactions;
- therefore, LiH is a useful benchmark case for separating "average learned placement bias" from "what the best retained structures actually need."

### L3: CH2 8q exposes branch-diverse low-energy solutions

The warm-start critical-structure analysis on `L3_CH2_Singlet_8q` shows a very different picture from Level 1.

Using the `0.00 mHa` bucket, 10 successful episodes were analysed after warm-start reconstruction from `first_hit_snapshot`. The main observations are:

- average original circuit size: about `44` gates;
- average retained circuit size after pruning: about `7.7` gates;
- average redundancy: about `78.5%`;
- retained structures are all different (`count=1` for every exact signature);
- the strongest anchor actions are local but not universal: `CNOT(0->4)`, `CNOT(1->5)`, and `CNOT(4->0)`.

This means Level 3 does **not** show a single dominant minimal motif. Instead, it supports multiple distinct low-energy branches, which is consistent with the near-degenerate landscape of `CH2 8q`.

### Why the L3 warm-start change mattered

The old cold-start reconstruction frequently failed on `CH2 8q`, because rebuilding the circuit from structure alone often dropped the optimizer into the wrong basin. After introducing `first_hit_snapshot` and warm-start reconstruction:

- the selected successful episodes can be rebuilt faithfully;
- pruning analysis becomes usable on harder cases;
- the benchmark can now distinguish between "one dominant motif" (Level 1) and "many branch-specific motifs" (Level 3).

### Current interpretation

At this stage, the critical-structure tool supports the following benchmark
story:

- `Level 1` is a compact, interpretable, motif-concentrated benchmark;
- `Level 3` is a branch-diverse, near-degenerate stability benchmark;
- `Level 6` is a family-style scalability benchmark built from the BeH2 basis / active-space expansion ladder;
- the difference between these two levels is therefore not just raw size, but the structure of the low-energy landscape itself.

## Level 6 Interpretation

`Level 6` should now be read as a **BeH2 family gradient**, not as one isolated single-molecule point.

The current Level-6 ladder is:

- `L6_BeH2_631G_8q`
- `L6_BeH2_6311G_10q`
- `L6_BeH2_CCPVDZ_12q`
- `L6_BeH2_CCPVDZ_14q`

This level is intended to probe:

- how RLQAS behaves as the BeH2 representation expands from `8q` to `14q`;
- how wall-clock cost and optimizer burden grow with system size;
- how search quality degrades as the Hamiltonian becomes denser and less local;
- whether the method still produces useful circuits under a realistic scalability ladder rather than a single fixed-size point.

So the benchmark meaning of `Level 6` is now:

- **not** "one hardest molecule"
- but **a structured scalability gradient on one molecular family**

## Full-Space Fingerprints

This section intentionally keeps only the unrestricted **Full-Space** fingerprint metrics. Physical-sector indicators are omitted here by design.

Columns:

- `Gap01`: `E1 - E0`
- `Hub`: qubit hub score
- `Asym`: qubit asymmetry
- `Z-only / XY-only / Mixed`: Pauli-weight fractions
- `>=4-body`: Pauli-weight fraction carried by terms with at least 4 Pauli factors
- `G1-G4`: Gershgorin-style matrix diagnostics

### Current Benchmark Suite

For `Level 6`, the rows below should be interpreted jointly as one scalability
family rather than as unrelated independent choices.

| Molecule | q | Gap01 | Hub | Asym | Z-only | XY-only | Mixed | >=4-body | G1 | G2 | BestDepth | Best Error (mHa) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `L1_BeH2_STO3G_6q` | 6 | 0.2120 | 1.1187 | 0.3560 | 0.9784 | 0.0216 | 0.0000 | 0.0216 | 0.9688 | 0.0937 | 2 | 0.269 |
| `L2_LiH_Equil_6q` | 6 | 0.0772 | 1.1349 | 0.2556 | 0.8759 | 0.0648 | 0.0594 | 0.0847 | 0.8906 | 0.2249 | 17 | 0.257 |
| `L3_CH2_Singlet_8q` | 8 | 0.0000 | 1.1616 | 0.3269 | 0.9067 | 0.0435 | 0.0499 | 0.0933 | 0.9453 | 0.2099 | 26 | 0.001 |
| `L4_H2O_StrongCorr_8q` | 8 | 0.0939 | 1.3888 | 0.6568 | 0.7235 | 0.1011 | 0.1754 | 0.2168 | 0.6484 | 0.0738 | 59 | 5.869 |
| `L5_H4_Chain_8q` | 8 | 0.2326 | 1.1188 | 0.3247 | 0.5989 | 0.1487 | 0.2524 | 0.4011 | 0.4180 | 0.0109 | 51 | 33.765 |
| `L6_BeH2_631G_8q` | 8 | 0.0895 | 1.1145 | 0.2283 | 0.9530 | 0.0470 | 0.0000 | 0.0470 | 0.9805 | 0.0462 | 50 | 0.116 |
| `L6_BeH2_6311G_10q` | 10 | 0.0627 | 1.0956 | 0.2531 | 0.9231 | 0.0349 | 0.0420 | 0.0769 | 0.9629 | 0.0105 | 45 | 0.248 |
| `L6_BeH2_CCPVDZ_12q` | 12 | 0.0677 | 1.1430 | 0.2538 | 0.8889 | 0.0303 | 0.0808 | 0.1111 | 0.9043 | 0.0192 | 63 | 1.243 |
| `L6_BeH2_CCPVDZ_14q` | 14 | 0.0615 | 1.1333 | 0.3765 | 0.7671 | 0.0520 | 0.1808 | 0.2329 | 0.6210 | 0.0003 | 91 | 5.008 |

`L6_BeH2_CCPVDZ_14q` also has its dense Hamiltonian stored in `mol_data`, so `G1-G2` can be computed directly from the same Gershgorin-style definitions used for the other rows.

The `BestDepth` and `Best Error (mHa)` columns are taken from the current `best_train.txt` outputs of the latest `crlqas` runs. The `L6_BeH2_631G_8q` row currently uses the latest available `LevelCheck` run, while the other rows come from the `Formal_EXP` runs.

### Active Candidate / Replacement Pool

These are the molecules that are currently being actively discussed as Level-3
or Level-4 alternatives.

| Molecule | q | Gap01 | Hub | Asym | Z-only | XY-only | Mixed | >=4-body | G1 | G2 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `L3_CH2_Singlet_R113_A080_6q` | 6 | 0.0107 | 1.1162 | 0.2021 | 0.9586 | 0.0414 | 0.0000 | 0.0414 | 0.9844 | 0.4997 |
| `L3_CH2_Singlet_R113_A130_6q` | 6 | 0.0000 | 1.1665 | 0.2450 | 0.8998 | 0.0468 | 0.0534 | 0.0802 | 0.9375 | 0.2339 |
| `L3_CH2_Singlet_R130_A100_6q` | 6 | 0.0000 | 1.1081 | 0.1959 | 0.9572 | 0.0428 | 0.0000 | 0.0428 | 0.9688 | 0.0777 |
| `L3_CH2_Singlet_R130_A130_6q` | 6 | 0.0000 | 1.1686 | 0.2554 | 0.8747 | 0.0532 | 0.0722 | 0.0967 | 0.9062 | 0.1852 |
| `L3_CH2_Singlet_8q` | 8 | 0.0000 | 1.1616 | 0.3269 | 0.9067 | 0.0435 | 0.0499 | 0.0933 | 0.9453 | 0.2099 |
| `L3_LiH_Stretch_6q` | 6 | 0.0000 | 1.2859 | 0.4704 | 0.7079 | 0.1236 | 0.1684 | 0.1756 | 0.5938 | 0.0774 |
| `L3_H3_Triangle_6q` | 6 | 0.0000 | 1.1436 | 0.2333 | 0.7895 | 0.1296 | 0.0809 | 0.2105 | 0.8125 | 0.4447 |
| `L4_H3_Linear_6q` | 6 | 0.0000 | 1.1462 | 0.2298 | 0.7091 | 0.1544 | 0.1366 | 0.2909 | 0.6562 | 0.0209 |

### CH2 6q -> 8q Expansion Note

To test whether the current `CH2` candidate is too easy mainly because of the small `CAS(2e,3o)` truncation, we also generated an `8q` version by enlarging the active space to `CAS(2e,4o)` while keeping the original geometry fixed.

Compared with `L3_CH2_Singlet_6q`, the new `L3_CH2_Singlet_8q` shifts in the expected "harder" direction:

- `Z-only`: `0.9615 -> 0.9067`
- `Mixed`: `0.0000 -> 0.0499`
- `>=4-body`: `0.0385 -> 0.0933`
- `Asym`: `0.2113 -> 0.3269`
- `G2`: `0.6335 -> 0.2099`

So even before running the new training sweep, the fingerprint view suggests that the `8q` active-space expansion is a more meaningful difficulty test than the original `6q` CH2 baseline.

### Additional Registry / Legacy Molecules

These are still part of the project registry and are useful as auxiliary references, but they are not the current main benchmark choices.

| Molecule | q | Gap01 | Hub | Asym | Z-only | XY-only | Mixed | >=4-body | G1 | G2 | G3 | G4 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `L1_H2_Equil_4q` | 4 | 0.6009 | 1.0353 | 0.0707 | 0.9045 | 0.0955 | 0.0000 | 0.0955 | 1.0000 | 1.3989 | 5.6730 | 0.3997 |
| `L1_BH_6q` | 6 | 0.0349 | 1.0292 | 0.0877 | 0.9565 | 0.0435 | 0.0000 | 0.0435 | 1.0000 | 2.3754 | 6.1132 | -0.1140 |
| `L2_BeH_Plus_4q` | 4 | 0.0000 | 1.1634 | 0.3545 | 0.8803 | 0.0431 | 0.0765 | 0.0431 | 1.0000 | 1.4930 | 25.9441 | -0.0535 |
| `L2_BF_8q` | 8 | 0.1153 | 1.1896 | 0.4449 | 0.8676 | 0.0244 | 0.1080 | 0.1324 | 0.8984 | 0.3400 | 15.3662 | -0.1064 |
| `L3_HeH_Plus_4q` | 4 | 0.0000 | 1.2260 | 0.4990 | 0.8373 | 0.0720 | 0.0907 | 0.0720 | 0.8750 | 0.9875 | 39.4561 | -0.0745 |
| `L5_H3_Linear_6q` | 6 | 0.0000 | 1.1462 | 0.2298 | 0.7091 | 0.1544 | 0.1366 | 0.2909 | 0.6562 | 0.0209 | 2.8517 | -0.8334 |
| `L6_BeH2_10q` | 10 | 0.2060 | 1.2149 | 0.4362 | 0.8893 | 0.0410 | 0.0698 | 0.1107 | 0.8730 | 0.0200 | 18.2063 | -0.2819 |
