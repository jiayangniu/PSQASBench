# Benchmark Molecule Decisions
---

## 1. Current Working Choices

| Level | Working molecule | Internal key / file family | Status |
|------|------|------|------|
| L1 Minimalism | BeH2 (STO-3G, 6 qubits) | `L1_BeH2_STO3G_6q` | temporarily fixed |
| L2 Asymmetry / Interaction Hubs | LiH (Equil., 6 qubits) | `L2_LiH_Equil_6q` | temporarily fixed, still needs difficulty validation |
| L3 Degeneracy / Stability | CH2 singlet (JW, 6 qubits) | `L3_CH2_Singlet_6q` | temporarily fixed |
| L4 Representation / Correlation | H2_Stretch (main) + stretched H2O (supplementary) | `L4_H2_Stretch_4q`, `L4_H2O_StrongCorr_8q` | temporarily fixed, still needs difficulty validation |
| L5 Topology | H4 Chain (8 qubits) | `L5_H4_Chain_8q` | temporarily fixed |
| L6 Scalability | BeH2 ladder (8q / 10q / 12q) | `L6_BeH2_631G_8q`, `L6_BeH2_6311G_10q`, `L6_BeH2_CCPVDZ_12q` | temporarily fixed |

Status labels used here:

- `temporarily fixed`: after simple runs / preliminary checks, we are comfortable using this molecule as the current working choice for the level.
- `temporarily fixed, still needs difficulty validation`: the molecule is the current working choice, but it still needs formal benchmark experiments to confirm that its difficulty profile really matches the intended level.

Current common setting for all currently fixed levels:

- Mapping: `Jordan-Wigner`
- Main benchmark task: `full Hamiltonian`
- If chemistry interpretation is discussed, we should explicitly state whether it is `without charge or spin restriction`

---

## 2. Common Generation Pipeline

For the current main benchmark, molecular Hamiltonians are generated with PennyLane `qchem.molecular_hamiltonian(...)`, with the following explicit controls:

- `charge`
- `mult`
- `active_electrons`
- `active_orbitals`
- `mapping='jordan_wigner'`

The current full-space generation scripts then:

1. Build the qubit Hamiltonian in the full `2^n` Hilbert space.
2. Separate the identity term into `energy_shift`.
3. Save the full Hamiltonian matrix, Pauli coefficients, Pauli strings, and full-space eigenvalues into `mol_data/*.npz`.

---

## 3. Level 1: BeH2 (STO-3G, 6 qubits)

### 3.1 Why it is currently selected

Our intended L1 selection criterion is not simply "small molecule" or "few qubits". The real goal of this level is to test how different methods respond when we tighten the circuit-depth budget.

More specifically, the desired L1 behavior is:

- when depth is not restricted, all benchmark methods should be able to reach chemical accuracy;
- once depth is restricted, the methods should begin to separate and show visibly different error profiles.

So L1 should be easy enough to solve in the unrestricted setting, but not so trivial that every method reaches numerically meaningless precision, and not so hard that all methods fail before the depth sensitivity can be studied.

Under this criterion, we excluded two earlier L1 candidates:

- `H2 (Equil.)` is too easy. Different methods can often drive the error down to around `1e-10`, which is far beyond the meaningful chemistry target and leaves almost no useful room to compare depth sensitivity.
- `BH` is too hard in the current benchmark representation. In current runs, the methods tend to stall around `error = 9.94 mHa`, so even without the intended depth restriction they already fail to cleanly reach chemical accuracy. That makes it a poor L1 anchor for studying depth-limited degradation.

Given these two exclusions, `BeH2_STO3G_6q` is currently the most practical L1 choice: it is intended to sit between an over-trivial case like `H2 (Equil.)` and an over-difficult case like `BH`, so that the error-vs-depth behavior can be meaningfully observed.

So the current working decision is:

- `L1 = L1_BeH2_STO3G_6q`

### 3.2 How this molecule is generated

Source:

- `mol_gen/prepare_beh2_basis_series.py`
- saved file family: `mol_data/L1_BeH2_STO3G_6q_*.npz`

Generation settings confirmed from the saved `.npz` metadata:

- Molecule: neutral linear `BeH2`
- Basis: `STO-3G`
- Mapping: `Jordan-Wigner`
- `active_electrons = 2`
- `active_orbitals = 3`
- `n_qubits = 6`
- `mult = 1` (singlet)

Geometry:

- `Be (0, 0, 0)`
- `H (0, 0, 1.326)`
- `H (0, 0, -1.326)`

Interpretation note:

- This is an aggressively truncated active-space model of neutral `BeH2`, not a near-full representation of the molecule.
- The current saved metadata records `basis`, `active_electrons`, and `active_orbitals`, but does not fully document the exact orbital-selection rationale beyond that truncation.

### 3.3 Molecules currently excluded from L1

#### Excluded: `H2 (Equil.)`

Reason:

- Too easy to serve as the only L1 anchor.
- As a sanity-check molecule it is still useful, but as a benchmark anchor it under-separates methods.

#### Excluded: `BH`

Reason:

- The chemistry story is awkward relative to the final `6-qubit` representation.
- Current experiments suggest both benchmark methods fail to reach chemical accuracy reliably enough for a clean `Minimalism` level.

---

## 4. Level 2: LiH (Equil., 6 qubits)

### 4.1 Why it is currently selected

The current working decision is:

- `L2 = LiH (Equil.)`

For L2, we do not treat a superficial `4-6-8` qubit progression as a design goal. That kind of progression is not especially meaningful for this level. The real criterion is whether the selected molecule cleanly represents `asymmetry / interaction-hub` difficulty without bringing in stronger confounding factors.

Under this criterion, `LiH (Equil.)` is the most practical L2 anchor:

- it is a neutral system,
- the asymmetry / hub-style narrative is already visible,
- it avoids unnecessary charged-system complications,
- it avoids mixing in heavier higher-order structure,
- and `LiH-6` is already a commonly used small benchmark setting in the literature, so the overall story is easier to communicate.

By contrast, the two excluded candidates each introduce extra difficulty dimensions that are not the main target of L2:

- `BeH+` mixes asymmetry with both charged-system concerns and degeneracy-related effects.
- `BF` mixes asymmetry with stronger `high-order / mixed structure` and also starts to pull the level toward a scalability-style story.

This choice is relatively stable, but it still needs explicit empirical validation of benchmark difficulty.

### 4.2 How this molecule is generated

Source:

- `mol_gen/prepare_molecules.py`
- physical-sector counterpart: `mol_gen/prepare_molecules_physical.py`
- benchmark registry key: `L2_LiH_Equil_6q`

Generation settings:

- Molecule: neutral `LiH`
- Mapping: `Jordan-Wigner`
- `charge = 0`
- `mult = 1`
- `active_electrons = 2`
- `active_orbitals = 3`
- `n_qubits = 6`

Geometry:

- `Li (0, 0, 0)`
- `H (0, 0, 1.595)`

Interpretation note:

- This is a clearly truncated active-space model of `LiH`, rather than a full-orbital representation.
- Even so, among the current L2 candidates it gives the cleanest asymmetry-focused benchmark story.

### 4.3 Molecules currently excluded from L2

#### Excluded: `BF`

Reason:

- Although its asymmetry signal is stronger, it also mixes in stronger `high-order / mixed structure`.
- It also starts to push the level toward a scalability-flavored story, which is not the point of L2.
- As a result, it is less clean as a pure `Asymmetry / Interaction Hubs` anchor.

#### Excluded: `BeH+`

Reason:

- It is a charged system.
- It also mixes in degeneracy-related effects.
- So it is less clean than `LiH (Equil.)` as a dedicated L2 anchor.

---

## 5. Level 3: CH2 Singlet (JW, 6 qubits)

### 5.1 Why it is currently selected

The current working decision is:

- `L3 = CH2 singlet, 6 qubits, Jordan-Wigner`

The main internal reason for selecting `CH2` is still benchmark design:

- among the current L3 candidates, it remains the strongest `near-degeneracy / stability-sensitive` case;
- it is more attractive than `LiH Stretch` as a benchmark anchor because its story is less dominated by stretch-induced pathologies;
- it gives us a chemically recognizable molecule while still remaining small enough for the current benchmark setting.

This judgment is also consistent with the current fingerprint table for the L3 candidates. We do not want to tie the final selection to exact metric values, since some of those definitions may still be refined. But at a qualitative level, the current table is still useful: it suggests that `CH2` is one of the cleanest L3 candidates, with a more "degeneracy / stability" flavor and less contamination from other difficulty sources than the main alternatives.

In addition, we now also have useful external support for this choice. In PennyLane's January 23, 2024 blog post [Top 20 molecules for quantum computing](https://pennylane.ai/blog/2024/01/top-20-molecules-for-quantum-computing), methylene (`CH2`) is explicitly included as one of the recommended molecules for quantum computing. The article highlights that deciding the spin character of methylene involves computing the singlet-triplet gap, which supports the idea that `CH2` is a meaningful electronically nontrivial benchmark molecule rather than an arbitrary in-house choice.

That said, this external support should be used carefully:

- it strengthens the case for choosing `CH2` as the L3 anchor;
- but it should not be presented as proving that our exact `JW 6-qubit, 2e/3o` model is identical to the full chemistry problem emphasized in the article.

So the cleanest position is:

- internally, `CH2` is chosen because it is our best current L3 benchmark candidate;
- externally, the PennyLane recommendation helps show that `CH2` is also a recognized molecule of independent interest for quantum-computing chemistry.

### 5.2 How this molecule is generated

Source:

- `mol_gen/prepare_molecules.py`
- physical-sector counterpart: `mol_gen/prepare_molecules_physical.py`
- benchmark registry key: `L3_CH2_Singlet_6q`

Generation settings:

- Molecule: neutral `CH2`
- Intended state: singlet (`mult = 1`)
- Mapping: `Jordan-Wigner`
- `charge = 0`
- `active_electrons = 2`
- `active_orbitals = 3`
- `n_qubits = 6`

Geometry:

- `C (0, 0, 0)`
- `H (0, 0.86, 0.73)`
- `H (0, -0.86, 0.73)`

Equivalent geometric summary:

- `C-H ≈ 1.128 A`
- `H-C-H angle ≈ 99.35 deg`

### 5.3 Physical interpretation and analysis plan

This point needs to be stated carefully.

What is physically reasonable:

- The bent singlet geometry is chemically plausible.
- The script indeed generates the Hamiltonian with `mult = 1`.
- The physical-sector version enforces `n_alpha = 1`, `n_beta = 1`.

What is not fully physical:

- The active-space truncation is very aggressive: `2e / 3o`.
- Therefore this is better described as a `CH2-derived active-space singlet model` rather than a near-complete representation of neutral `CH2`.

What we treat as the main experiment:

- the main L3 benchmark remains `JW 6-qubit CH2` in the current full-Hamiltonian benchmark pipeline.

What we treat as analysis experiments rather than the main task:

- `full-space` vs `restricted sector`
- `JW` vs `tapered` or other reduction-based variants

Why these are analysis experiments:

- `restricted sector` is useful for discussing physical correctness and for checking how much of the apparent degeneracy survives after enforcing the intended electron/spin constraints;
- `tapering` is useful for discussing representation reduction and encoding sensitivity;
- but neither should replace the main L3 experiment, because they modify the interpretation or representation of the original benchmark task.

What we observed from the physical-sector diagnostic:

- Full-space lowest gap: `0.0 Ha`
- Restricted-sector lowest gap: about `0.0185 Ha`
- Sector size for the singlet diagnostic: `9` basis states inside the full `64`-dimensional Hilbert space

Interpretation:

- The exact full-space degeneracy is not fully physical.
- But after restricting to the correct singlet sector, `CH2` still remains a strong `near-degeneracy` case.
- Therefore it is still a reasonable `L3` anchor, as long as the paper does not overclaim it as an exactly physical degeneracy benchmark in the unrestricted full-space sense.
- The most natural presentation is to keep `JW 6-qubit CH2` as the main benchmark case, and then discuss `restricted sector` and possibly `tapered` variants as follow-up analysis for physical correctness and encoding sensitivity.

### 5.4 Molecules currently excluded from L3

#### Excluded: `HeH+`

Reason:

- Charged system.
- In the current fingerprint summary, it does not look as cleanly focused on the intended L3 story as `CH2`.
- Less attractive than a neutral molecule for the main `L3` anchor.

#### Excluded: `LiH Stretch`

Reason:

- In the current fingerprint summary, it looks more mixed and less cleanly centered on the intended L3 story than `CH2`.
- It is still useful as a stability / sector-sensitivity reference.
- But its story is more heavily entangled with bond stretching effects, and therefore less clean than `CH2` as the main anchor.

#### Excluded: `H3+ (Triangle)`

Reason:

- In the current fingerprint summary, it also looks more mixed than `CH2` as an L3 anchor.
- Interesting, but the charged-system interpretation is less clean.
- Less aligned with the current decision to keep the main benchmark narrative as simple as possible.

---

## 6. Level 4: H2 Stretch (main) + stretched H2O (supplementary)

### 6.1 Why they are currently selected

The current working decision is:

- `L4 main anchor = H2_Stretch`
- `L4 supplementary larger case = stretched H2O`

The main purpose of L4 is to study how ground-state entanglement and representation burden challenge different methods. Following the coauthor's recommendation, this level is no longer justified primarily by `high-order ratio / k-locality`. Instead, the guiding idea is to look more directly at exact-ground-state entanglement.

Under that framing, `H2_Stretch` is currently the cleanest L4 anchor:

- it gives the strongest and most visually clear entanglement signal among the current small candidates;
- it is chemically simple, so the interpretation is easy to explain;
- it avoids the charged-system and sector complications that make some other candidates harder to interpret;
- and it works well as a direct stress test of entanglement demand on the method itself.

At the same time, we do not want L4 to look artificially tiny, so we also keep one larger correlated case:

- stretched `H2O` remains in L4 as a supplementary example of a larger representation-sensitive molecule;
- but it is not the main anchor, because using `8 qubits` alone as the only L4 definition may make this level look harder than intended.

### 6.2 Current thoughts and concerns

The current L4 decision reflects the following ideas and concerns:

- The core scientific question for this level is the challenge created by entanglement requirement, not just generic chemistry complexity.
- `H2_Stretch` looks especially attractive because its entanglement behavior is very strong and very clean under the current exact-ground-state analysis.
- The main concern with `H2_Stretch` is size: it is only a `4-qubit` system.
- Ideally, we would also like a clean `6-qubit` molecule for this level, but we have not yet found one that is both physically reasonable and interpretation-friendly.
- `H3_Linear` is attractive from a size perspective, but it currently brings in degeneracy / sector concerns and therefore is not clean enough to serve as the sole main anchor.
- `N2` and `C2` are conceptually interesting for correlation studies, but at `6 qubits` they would likely require truncation that is too aggressive for a clean benchmark story.
- `H2O` became much more plausible after switching to a stretched geometry, but there is still a concern that an `8-qubit` `H2O` used by itself could make L4 too difficult.

So the current compromise is:

- use `H2_Stretch` as the canonical L4 anchor;
- keep stretched `H2O` as a secondary larger case that supports the same level narrative without redefining the level by size alone.

### 6.3 How these molecules are generated

Source:

- `mol_gen/prepare_molecules.py`
- physical-sector counterpart: `mol_gen/prepare_molecules_physical.py`

Main anchor: `L4_H2_Stretch_4q`

- Molecule: neutral `H2`
- Mapping: `Jordan-Wigner`
- `charge = 0`
- `mult = 1`
- `active_electrons = 2`
- `active_orbitals = 2`
- `n_qubits = 4`
- Geometry:
  - `H (0, 0, 0)`
  - `H (0, 0, 2.5)`

Supplementary larger case: `L4_H2O_StrongCorr_8q`

- Molecule: neutral stretched `H2O`
- Mapping: `Jordan-Wigner`
- `charge = 0`
- `mult = 1`
- `active_electrons = 4`
- `active_orbitals = 4`
- `n_qubits = 8`
- Geometry:
  - `O (0, 0, 0)`
  - `H (0, 1.186, 0.918)`
  - `H (0, -1.186, 0.918)`

Interpretation note:

- `H2_Stretch` is the cleanest small entanglement-stress case.
- stretched `H2O` is retained to show that the same level idea can also appear in a larger molecule, but it should be interpreted as a supplementary case rather than the sole defining anchor.

### 6.4 Molecules currently not used as the main L4 anchor

#### Not used as the main anchor: `H3_Linear`

Reason:

- It is attractive because it sits at `6 qubits`.
- But it currently mixes in degeneracy and sector-related concerns.
- So it is not as clean as `H2_Stretch` for a primary entanglement-focused L4
  story.

#### Not used as the main anchor: `H2O` alone

Reason:

- The stretched geometry makes it much more plausible than the old equilibrium
  version.
- But if used alone, it may make L4 look harder than intended.
- It is therefore better kept as a supplementary larger case.

#### Deferred rather than adopted: `N2` or `C2` at 6 qubits

Reason:

- Both are conceptually appealing from a correlation point of view.
- But at `6 qubits` they would likely require active-space truncation that is
  too aggressive for the current benchmark redesign.

---

## 7. Level 5: H4 Chain (8 qubits)

### 7.1 Why it is currently selected

The current working decision is:

- `L5 = H4 Chain`

The main point of L5 is not generic molecular difficulty. It is to test topology pressure: how much the method is affected when the useful entangling pattern is constrained by routing and connectivity.

Under this criterion, `H4 Chain` is the cleanest current L5 anchor:

- it is more naturally aligned with a one-dimensional nearest-neighbor topology story;
- it creates clearer routing pressure than `H3 Linear`;
- it avoids unnecessary charged-system complications;
- and it is more distinct from the current L4 discussion than reusing another `H3` case as the main topology anchor.

So the most natural L5 interpretation is:

- `H4 Chain` is the main topology benchmark;
- the intended difficulty comes from routing and connectivity pressure rather than from making the chemistry story itself more complicated.

### 7.2 How this molecule is generated

Source:

- `mol_gen/prepare_molecules.py`
- benchmark registry key: `L5_H4_Chain_8q`

Generation settings:

- Molecule: neutral `H4`
- Mapping: `Jordan-Wigner`
- `charge = 0`
- `mult = 1`
- `active_electrons = 4`
- `active_orbitals = 4`
- `n_qubits = 8`

Geometry:

- `H (0, 0, 0)`
- `H (0, 0, 1.0)`
- `H (0, 0, 2.0)`
- `H (0, 0, 3.0)`

Interpretation note:

- This is a very simple and highly structured molecular family, which is exactly why it works for L5.
- The goal here is not chemistry richness, but a clean topology-sensitive benchmark.

### 7.3 Molecules currently excluded from L5

#### Excluded: `H3 Linear`

Reason:

- It is a charged system.
- It overlaps too much with the current `H3`-based representation discussion used elsewhere.
- It is less compelling than `H4 Chain` as the single anchor for routing / topology pressure.

---

## 8. Level 6: BeH2 Scalability Ladder

### 8.1 Why it is currently selected

The current working decision is:

- `L6 main ladder = BeH2 8q / 10q / 12q`
- `L6 optional extension = BeH2 14q`

The point of L6 is scalability, so the main design goal is to change problem size while keeping the molecular family and overall chemistry story as stable as possible.

That is why `BeH2` remains the right L6 backbone:

- it gives a same-family ladder instead of mixing unrelated molecules;
- it lets us discuss scaling without changing the benchmark identity at every rung;
- and it keeps the L6 story cleaner than a design that changes both molecule and size simultaneously.

The current main ladder is:

- `L6_BeH2_631G_8q`
- `L6_BeH2_6311G_10q`
- `L6_BeH2_CCPVDZ_12q`

Two additional design choices are important here:

- `L1_BeH2_STO3G_6q` is no longer part of the L6 main ladder, because it now serves as the L1 anchor.
- `14q` should remain an optional extension rather than a required main benchmark rung, because it increases generation and training uncertainty too early.

So the cleanest L6 presentation is:

- main benchmark: `BeH2 8q / 10q / 12q`
- optional stretch goal: `BeH2 14q`

### 8.2 How these molecules are generated

Primary source:

- `mol_gen/prepare_beh2_basis_series.py`

Current main-rung file families:

- `L6_BeH2_631G_8q`
- `L6_BeH2_6311G_10q`
- `L6_BeH2_CCPVDZ_12q`

Shared molecular setting:

- Molecule: neutral linear `BeH2`
- Mapping: `Jordan-Wigner`
- `charge = 0`
- `mult = 1`
- same equilibrium geometry across the ladder

Geometry:

- `Be (0, 0, 0)`
- `H (0, 0, 1.326)`
- `H (0, 0, -1.326)`

Current main-rung metadata from the saved `.npz` files:

- `6-31G`: `active_electrons = 2`, `active_orbitals = 4`, `n_qubits = 8`
- `6-311G`: `active_electrons = 2`, `active_orbitals = 5`, `n_qubits = 10`
- `cc-pVDZ`: `active_electrons = 2`, `active_orbitals = 6`, `n_qubits = 12`

Interpretation note:

- This is still an active-space benchmark ladder rather than a full-basis chemistry ladder.
- Even so, it is currently the cleanest internal way to test scalability while keeping the molecule family fixed.

### 8.3 Molecules or variants currently excluded from the L6 main ladder

#### Excluded from the main ladder: `L1_BeH2_STO3G_6q`

Reason:

- It now serves as the L1 anchor.
- Reusing it inside the main L6 ladder would blur the separation between `Minimalism` and `Scalability`.

#### Excluded from the main commitment: `BeH2 14q`

Reason:

- It is still valuable as an extension target.
- But it should not be written into the main benchmark commitment until generation and resource cost are more stable.

#### De-emphasized: legacy single-case `L6_BeH2_Scalability_10q`

Reason:

- The new L6 story is a multi-rung `BeH2` ladder rather than a single 10-qubit case.
- The old 10q scalability case can remain in the repo, but it should no longer define the main L6 benchmark narrative by itself.

---

## 9. Explicitly Deferred Follow-up Checks

These choices are working decisions, not final chemistry claims. The following checks are still needed:

1. Validate that `L1 = BeH2_STO3G_6q` really behaves like a usable `Minimalism` case in experiment.
2. Validate the actual algorithmic difficulty of `L2 = LiH (Equil.)`.
3. Keep `L3 = CH2` as the main anchor, but add a supplement or appendix comparison:
   - `full-space JW` vs `restricted sector`
4. If resources permit, later add an encoding/reduction variant for L3:
   - `JW` vs `parity / tapered`
5. Keep checking whether any clean `6-qubit` L4 candidate emerges that is more convincing than the current compromise.
6. Keep `L6 14q` as a stretch target until data generation and training stability are confirmed.

---

## 10. Short Version For Paper Drafting

If we need one compact internal summary sentence for now, it is:

> Current working choices are `L1 = BeH2_STO3G_6q`, `L2 = LiH (Equil.)`, `L3 = JW 6-qubit CH2 singlet`, `L4 = H2_Stretch` with stretched `H2O` as a supplementary larger case, `L5 = H4 Chain`, and `L6 = BeH2 8q / 10q / 12q` with `14q` as an optional extension; all are active-space models under Jordan-Wigner mapping.
