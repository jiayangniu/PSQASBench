# PSQASBench Paper Structure

> Internal writing blueprint for the NeurIPS 2026 paper.
> This file reflects the current working benchmark decisions, not an archival record of old alternatives.

---

## 1. Paper in One Sentence

**PSQASBench is a principled benchmark for RL-based quantum architecture search that replaces ad hoc molecule choice, ambiguous evaluation timing, and single-metric reporting with a diagnostic 6-level molecular suite, a unified periodic-evaluation protocol, and multi-objective policy diagnostics.**

---

## 2. Core Claims

The paper should make four tightly connected claims:

1. **Evaluation in RL-QAS is currently inconsistent.**
   Different papers test on different molecules, use different checkpoint conventions, and report different success criteria, so method comparisons are not reliable.

2. **A benchmark for RL-QAS must be diagnostic, not just large.**
   The point is not to collect many chemistry instances, but to cover qualitatively different failure modes of circuit search policies.

3. **PSQASBench provides that benchmark.**
   It defines a fixed 6-level molecular suite, a unified training/evaluation protocol, and benchmark outputs that preserve both final quality and training dynamics.

4. **The benchmark reveals failure modes that standard reporting hides.**
   In particular, methods may reach low energy with poor circuit efficiency, may look stable under final-error reporting but unstable under periodic policy evaluation, and may scale poorly even within a single molecule family.

---

## 3. Final Benchmark Identity

### 3.1 Main Task Definition

Current benchmark identity:

- active-space molecular Hamiltonians
- Jordan-Wigner mapping
- noiseless setting
- full Hilbert-space task definition
- periodic policy evaluation during training

Important wording choice for the paper:

- the main benchmark task is **full-space**
- physical-sector analysis is **supplementary validity analysis**, not the main task definition

This lets us stay honest about what the current methods actually optimize, while still discussing chemistry realism where needed.

### 3.2 Current 6-Level Suite

| Level | Role | Main benchmark case | Notes |
|------|------|------|------|
| L1 | Minimalism | `L1_BeH2_STO3G_6q` | replaces over-trivial `H2` and over-hard `BH` |
| L2 | Asymmetry / interaction hub | `L2_LiH_Equil_6q` | cleanest current neutral asymmetry case |
| L3 | Degeneracy / stability | `L3_CH2_Singlet_8q` | near-degeneracy / branch-sensitive case with stronger structure-search burden |
| L4 | Representation / correlation | `L4_H2O_StrongCorr_8q` | main 8-qubit correlated anchor |
| L5 | Topology / routing pressure | `L5_H4_Chain_8q` | intended for connectivity-sensitive experiments |
| L6 | Scalability ladder | `L6_BeH2_631G_8q`, `L6_BeH2_6311G_10q`, `L6_BeH2_CCPVDZ_12q` | same-family scaling story |

Optional extension:

- `L6_BeH2_CCPVDZ_14q` as a stretch target, not part of the core required ladder

### 3.3 Benchmark Philosophy

Each level should correspond to a **different diagnostic pressure**, not merely a different qubit count.

- L1 asks whether a policy can avoid unnecessary structure.
- L2 asks whether the policy allocates entangling resources asymmetrically when the Hamiltonian is asymmetric.
- L3 asks whether the policy remains stable under near-degeneracy / small-gap pressure.
- L4 asks whether the method handles strong entanglement / correlation burden.
- L5 asks whether routing and connectivity constraints damage the search policy.
- L6 asks whether the method scales when the molecular family is held fixed and only size is increased.

This should be the paper's central justification for the dataset design.

---

## 4. Unified Evaluation Protocol

### 4.1 What We Standardize

The benchmark standardizes three things:

1. **What to test on**
   Fixed 6-level diagnostic suite.

2. **When to evaluate**
   Periodic evaluation during training, rather than ambiguous “last checkpoint only”.

3. **What to measure**
   Both performance and circuit quality, plus policy-diversity diagnostics.

### 4.2 Training-Time Evaluation

Current intended protocol:

- multiple random seeds per method/molecule
- periodic evaluation every fixed number of training episodes
- each evaluation uses `K` greedy rollouts from the current policy
- report both training-best and eval-best information

What should be emphasized in the paper:

- we do **not** rely on a vague “chosen checkpoint after the fact”
- we evaluate policies **during training** under a fixed schedule
- this makes policy quality and training stability observable

### 4.3 Reported Outputs

The benchmark should explicitly say that each run preserves:

- run-level metadata
- episode-level training summaries
- episode-level traces
- policy-loss history
- best training circuit summary
- best evaluation snapshot

This is part of the benchmark contribution because it directly improves reproducibility and post hoc analysis.

---

## 5. Main Metrics

### 5.1 Primary Performance View: Pareto, Not Single Number

The main reporting lens should be:

- **energy error**
- **CNOT count**
- **rotation count / depth** as supporting circuit-cost views

Core idea:

- a benchmark for architecture search should not reward unnecessarily deep circuits just because they eventually reach low energy
- therefore we report the quality-cost tradeoff, not energy alone

Recommended main plot family:

- Energy error vs CNOT count
- optionally Energy error vs depth for L1/L2

### 5.2 Scalar Summary Metrics

Recommended benchmark summaries:

- `SR@chem`
- `CNOT@chem`
- `best_error_mha`
- `mean_error_mha`
- `mean_cnots`
- wall-clock cost where relevant for scalability sections

### 5.3 Policy Circuit Diversity (PCD)

PCD should remain part of the benchmark, but the definition should now be stated as **state-based**, not unitary-based.

#### Final definition

Given `K` policy rollouts:

- `D_struct`: set all rotation angles to `pi/4`, generate the output state of each circuit, and compute the mean pairwise state-fidelity distance
- `D_func`: use the final optimized angles, generate the output state of each circuit, and compute the mean pairwise state-fidelity distance

Distance definition:

- `d(psi_i, psi_j) = 1 - |<psi_i | psi_j>|^2`

Why this is the right definition:

- it scales to larger qubit systems
- it stays faithful to the actual task, which is state preparation for VQE-like objectives
- it avoids the full-unitary blowup that makes the old definition impractical for larger systems

Interpretation table:

| D_struct | D_func | Interpretation |
|----------|--------|----------------|
| low | low | policy structurally stable and functionally stable |
| high | low | multiple structurally different but functionally similar solutions |
| low | high | structure stable but optimization unstable |
| high | high | policy behavior unreliable / random-walk-like |

Important nuance for the paper:

- topology-only similarity can be discussed as an optional auxiliary analysis
- but it should **not** replace state-based `D_struct` as the main diversity metric

---

## 6. Proposed Paper Structure

### Section 1. Introduction

Goal:

- establish that RL-QAS evaluation is broken in a specific, reproducible way

Main points:

- RL for QAS is increasingly popular, but cross-paper comparisons are unreliable
- current inconsistency appears along three axes:
  - molecule choice
  - evaluation timing
  - reported metric
- PSQASBench is proposed as a principled benchmark that addresses all three

Close the introduction with:

- dataset contribution
- evaluation-protocol contribution
- metric contribution
- empirical benchmark findings

### Section 2. Related Work

Subsections:

- RL for QAS methods
- existing chemistry benchmark choices in QAS papers
- benchmark design in adjacent quantum / ML areas

The goal is not a giant survey. The goal is to show:

- the field lacks a common benchmark
- current evaluation practice is fragmented

### Section 3. PSQASBench

This section should define the benchmark itself, not yet the findings.

Suggested subsections:

#### 3.1 Benchmark Design Principles

- diagnostic rather than generic dataset construction
- separate failure modes across levels
- same representation family where possible
- unified evaluation schedule

#### 3.2 The 6-Level Molecular Diagnostic Suite

Include:

- one compact table with `Level / role / main molecule / qubits / diagnostic target`
- one short paragraph per level explaining why this molecule is the current anchor

Important writing choice:

- do not overload this section with too much chemistry prose
- detailed per-molecule rationale can be moved to appendix/supplement if needed

#### 3.3 Benchmark Protocol

Include:

- training setting
- periodic evaluation schedule
- seeds
- noiseless assumption
- mapping choice
- full-space main task definition

#### 3.4 Metrics

Include:

- Pareto view
- scalar summaries
- state-based PCD

This subsection should explicitly explain why state-based PCD replaces the old full-unitary version for scalable benchmarking.

### Section 4. Benchmark Findings

This is the paper's empirical center.

Recommended structure:

#### 4.1 Finding 1: Circuit Structure Bias

Expected narrative:

- some RL methods can reach good energy only by using far more entangling structure than necessary
- this is visible in L1/L2 and is hidden by energy-only reporting

Main evidence:

- Pareto plots
- energy vs CNOT / depth plots

#### 4.2 Finding 2: Difficulty Is Multi-Source, Not Monotone in Qubit Count

Expected narrative:

- benchmark levels separate different failure pressures
- small systems can still be diagnostically hard
- size alone is not an adequate benchmark design principle

Main evidence:

- per-level comparison across methods
- selected case studies from L2/L3/L4/L5

#### 4.3 Finding 3: Policy Stability and Diversity Matter

Expected narrative:

- periodic evaluation reveals instability that final-error reporting would hide
- `D_struct / D_func` help distinguish “many equivalent solutions” from “optimization instability”

Main evidence:

- evaluation curves over training
- PCD trend examples on representative cases

#### 4.4 Finding 4: Same-Family Scaling Still Breaks Methods

Expected narrative:

- within the BeH2 ladder, scaling pressure becomes visible even without changing molecule family
- L6 therefore isolates scalability more cleanly than a mixed-molecule large-case evaluation

Main evidence:

- BeH2 `8q / 10q / 12q`
- `14q` as optional extension if stable enough

### Section 5. Case Study or Proof-of-Concept Improvement

If the current plan remains `CRLQAS-STOP`, this section should be:

- a focused case study showing how a benchmark-diagnosed failure mode can guide method repair

The point is not to introduce a whole new benchmark paper inside this one.
The point is:

- PSQASBench is not only diagnostic
- it can guide method design

### Section 6. Discussion and Limitations

Must include:

- benchmark is full-space by design; sector-aware analysis is supplementary
- benchmark levels are active-space models, not exact full-chemistry surrogates
- L6 `14q` remains an extension, not yet the core requirement
- topology-only metrics are not used as the main diversity metric
- future work: sector-aware benchmark variants, noise-aware extensions, larger scaling ladder

### Section 7. Conclusion

Keep this short and crisp:

- restate benchmark problem
- restate PSQASBench solution
- restate what kinds of failures the benchmark reveals

---

## 7. Main Figures to Prioritize

Because time is limited, the paper should prioritize a small number of high-yield figures.

### Must-Have

1. **Overview figure**
   The three benchmark problems:
   - what to test on
   - when to evaluate
   - what to measure

2. **6-level benchmark table**
   Level, molecule, qubits, diagnostic role.

3. **Pareto figure**
   Representative L1/L2 case showing energy-only reporting failure.

4. **Policy-evaluation trend figure**
   `best_error / SR / D_struct / D_func` over training on representative cases.

5. **Scalability figure**
   BeH2 ladder `8q / 10q / 12q`, optionally `14q`.

### Nice-to-Have

6. **Physical-sector supplementary figure**
   Especially for `CH2`, showing why full-space remains the benchmark task but sector analysis matters for interpretation.

7. **Case-study improvement figure**
   If `CRLQAS-STOP` remains in the paper.

---

## 8. Minimum Experimental Story Needed for Submission

To make the paper coherent, we do **not** need every possible experiment.
We need the minimum set that validates the benchmark story.

Minimum package:

1. At least one clear L1/L2 example showing circuit-structure bias.
2. At least one L3/L4 example showing instability or representation burden.
3. At least one L5 example showing topology sensitivity.
4. A clean L6 BeH2 ladder result on `8q / 10q / 12q`.
5. State-based PCD demonstrated on at least representative levels.

If time is tight:

- keep `14q` as optional
- keep physical-sector analysis concise
- keep the proof-of-concept method section short

---

## 9. Writing Rules

To keep the paper strong:

- do not claim chemical realism beyond what the active-space models support
- do not claim every fingerprint has predictive power before data show it
- do not mix “benchmark design rationale” with “empirical benchmark findings”
- do not let `14q` instability redefine the core benchmark commitment
- do not revert to energy-only reporting in the results section

---

## 10. Current Internal Summary

If we need one short drafting summary sentence, use this:

> PSQASBench is a full-space, noiseless, Jordan-Wigner benchmark for RL-based quantum architecture search built around a 6-level diagnostic molecular suite: `L1 = BeH2_STO3G_6q`, `L2 = LiH_Equil_6q`, `L3 = CH2_Singlet_6q`, `L4 = H2_Stretch_4q` with stretched `H2O_8q` as a supplementary larger case, `L5 = H4_Chain_8q`, and `L6 = BeH2 8q / 10q / 12q`, with `14q` retained as an optional extension; evaluation is performed periodically during training and policy diversity is measured with state-based `D_struct / D_func`.
