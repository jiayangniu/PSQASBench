# Benchmark Molecule Decisions

This document is the merged and current benchmark-design record for PSQASBench.
It replaces the older split between `BENCHMARK_MOLECULE_DECISIONS.md` and
`BENCHMARK_REDESIGN_TODO.md`.

The goal here is to record the benchmark choices that we are actually using,
plus the design rules that still matter for paper writing and experiment
interpretation. Old branches, abandoned candidates, and already-resolved TODO
items are intentionally removed.

---

## 1. Final 6-Level Benchmark Suite

| Level | Theme | Main benchmark choice | Internal key(s) |
|------|------|------|------|
| L1 | Minimalism / depth sensitivity | BeH2 (STO-3G, 6 qubits) | `L1_BeH2_STO3G_6q` |
| L2 | Asymmetry / interaction hubs | LiH (equilibrium, 6 qubits) | `L2_LiH_Equil_6q` |
| L3 | Near-degeneracy / stability | CH2 singlet (JW, 6 qubits) | `L3_CH2_Singlet_6q` |
| L4 | Representation / correlation burden | H2 stretch (main) | `L4_H2_Stretch_4q` |
| L4 supplementary | Larger correlated companion case | stretched H2O | `L4_H2O_StrongCorr_8q` |
| L5 | Topology / routing pressure | H4 chain (8 qubits) | `L5_H4_Chain_8q` |
| L6 | Scalability ladder | BeH2 ladder (8q / 10q / 12q) | `L6_BeH2_631G_8q`, `L6_BeH2_6311G_10q`, `L6_BeH2_CCPVDZ_12q` |
| L6 optional extension | Larger scalability stress test | BeH2 (14 qubits) | `L6_BeH2_CCPVDZ_14q` |

Short paper-ready summary:

> PSQASBench currently uses `L1 = BeH2_STO3G_6q`, `L2 = LiH_Equil_6q`,
> `L3 = CH2_Singlet_6q`, `L4 = H2_Stretch_4q` with stretched `H2O_8q` as a
> supplementary larger correlated case, `L5 = H4_Chain_8q`, and
> `L6 = BeH2 8q / 10q / 12q`, with `14q` treated as an optional extension.

---

## 2. Benchmark-Wide Design Rules

These choices are fixed across the current benchmark narrative.

### 2.1 Main representation

- Mapping: `Jordan-Wigner`
- Main task: `full-space Hamiltonian`
- Dataset type: `active-space molecular models`

This means the benchmark should be described as a full-space qubit-Hamiltonian
benchmark built from active-space molecular constructions, not as a
fully converged chemistry benchmark in a large orbital basis.

### 2.2 Physical-sector analysis

- The main benchmark remains the unrestricted full-space problem.
- Charge- or spin-restricted sector analysis is supplementary diagnosis, not a
  replacement for the main benchmark definition.

This matters most for L3: sector-aware analysis is useful for interpretation,
but the main benchmark case remains the standard full-space qubit Hamiltonian.

### 2.3 Mapping and qubit reduction

- `parity`, `tapering`, or other reduction-based encodings are not part of the
  main 6-level benchmark axis.
- They can be used later as side analyses on representative cases.

### 2.4 Difficulty interpretation

- L1 focuses on depth sensitivity, not just small qubit count.
- L4 is justified by representation / entanglement burden, not by
  `k-locality` or `>=4-body ratio` alone.
- L6 focuses on same-family scaling, not on mixing different molecules at
  different sizes.

---

## 3. Molecule-by-Molecule Decisions

### 3.1 L1: BeH2 (STO-3G, 6 qubits)

Chosen as the minimalism anchor because it sits between two unsatisfying
extremes:

- `H2 (equilibrium)` is too easy and under-separates methods.
- `BH` is too hard in the current representation and does not serve cleanly as
  a depth-limited benchmark anchor.

So L1 is:

- `L1_BeH2_STO3G_6q`

Generation source:

- `mol_gen/prepare_beh2_basis_series.py`

Key settings:

- neutral linear `BeH2`
- `basis = STO-3G`
- `active_electrons = 2`
- `active_orbitals = 3`
- `n_qubits = 6`
- `mult = 1`

Geometry:

- `Be (0, 0, 0)`
- `H (0, 0, 1.326)`
- `H (0, 0, -1.326)`

Interpretation note:

- This is a deliberately small active-space model used as a benchmark anchor,
  not a near-complete chemistry representation of `BeH2`.

### 3.2 L2: LiH (equilibrium, 6 qubits)

Chosen as the asymmetry / interaction-hub anchor because it gives a clean,
neutral, easy-to-explain benchmark without dragging in stronger confounds such
as charged-system interpretation or heavier mixed difficulty.

So L2 is:

- `L2_LiH_Equil_6q`

Generation source:

- `mol_gen/prepare_molecules.py`
- `mol_gen/prepare_molecules_physical.py` for sector-aware diagnosis

Key settings:

- neutral `LiH`
- `active_electrons = 2`
- `active_orbitals = 3`
- `n_qubits = 6`
- `mult = 1`

Geometry:

- `Li (0, 0, 0)`
- `H (0, 0, 1.595)`

### 3.3 L3: CH2 singlet (JW, 6 qubits)

Chosen as the near-degeneracy / stability anchor because it is the cleanest
current case for that role while remaining small and chemically recognizable.

So L3 is:

- `L3_CH2_Singlet_6q`

Generation source:

- `mol_gen/prepare_molecules.py`
- `mol_gen/prepare_molecules_physical.py` for sector-aware diagnosis

Key settings:

- neutral `CH2`
- intended singlet: `mult = 1`
- `active_electrons = 2`
- `active_orbitals = 3`
- `n_qubits = 6`

Geometry:

- `C (0, 0, 0)`
- `H (0, 0.86, 0.73)`
- `H (0, -0.86, 0.73)`

Interpretation note:

- The main benchmark is still the full-space JW Hamiltonian.
- Sector-restricted analysis is useful for interpretation, but it does not
  redefine the benchmark task.
- This should be described as a `CH2-derived active-space singlet model`, not
  as a near-complete representation of neutral methylene.

### 3.4 L4: H2 stretch as main anchor, stretched H2O as supplementary case

L4 is now defined by representation / entanglement burden. It is no longer
framed primarily through high-order Pauli statistics.

The main anchor is:

- `L4_H2_Stretch_4q`

The supplementary larger correlated case is:

- `L4_H2O_StrongCorr_8q`

Generation source:

- `mol_gen/prepare_molecules.py`
- `mol_gen/prepare_molecules_physical.py`

Main anchor settings:

- neutral `H2`
- `active_electrons = 2`
- `active_orbitals = 2`
- `n_qubits = 4`
- stretched bond:
  - `H (0, 0, 0)`
  - `H (0, 0, 2.5)`

Supplementary case settings:

- stretched neutral `H2O`
- `active_electrons = 4`
- `active_orbitals = 4`
- `n_qubits = 8`
- geometry:
  - `O (0, 0, 0)`
  - `H (0, 1.186, 0.918)`
  - `H (0, -1.186, 0.918)`

Interpretation note:

- `H2_Stretch` is the canonical L4 anchor because it gives the cleanest small
  entanglement-stress example.
- stretched `H2O` is kept to show the same level idea in a larger correlated
  system, but it is not the sole defining anchor.

### 3.5 L5: H4 chain (8 qubits)

Chosen as the topology / routing-pressure anchor because it gives a clean,
neutral, one-dimensional structure whose main difficulty is connectivity
pressure rather than chemistry richness.

So L5 is:

- `L5_H4_Chain_8q`

Generation source:

- `mol_gen/prepare_molecules.py`

Key settings:

- neutral `H4`
- `active_electrons = 4`
- `active_orbitals = 4`
- `n_qubits = 8`

Geometry:

- `H (0, 0, 0)`
- `H (0, 0, 1.0)`
- `H (0, 0, 2.0)`
- `H (0, 0, 3.0)`

### 3.6 L6: BeH2 scalability ladder

Chosen because scalability should be tested by increasing problem size while
keeping the molecular family fixed.

Main ladder:

- `L6_BeH2_631G_8q`
- `L6_BeH2_6311G_10q`
- `L6_BeH2_CCPVDZ_12q`

Optional extension:

- `L6_BeH2_CCPVDZ_14q`

Generation source:

- `mol_gen/prepare_beh2_basis_series.py`

Shared settings:

- neutral linear `BeH2`
- same equilibrium geometry across the ladder
- `mult = 1`

Geometry:

- `Be (0, 0, 0)`
- `H (0, 0, 1.326)`
- `H (0, 0, -1.326)`

Current ladder metadata:

- `6-31G`: `active_electrons = 2`, `active_orbitals = 4`, `n_qubits = 8`
- `6-311G`: `active_electrons = 2`, `active_orbitals = 5`, `n_qubits = 10`
- `cc-pVDZ`: `active_electrons = 2`, `active_orbitals = 6`, `n_qubits = 12`
- `cc-pVDZ extension`: `active_electrons = 2`, `active_orbitals = 7`, `n_qubits = 14`

Interpretation note:

- This is an active-space scalability ladder rather than a fully converged
  basis-set ladder.
- `L1_BeH2_STO3G_6q` is intentionally not part of the L6 main ladder because it
  now serves as the L1 anchor.

---

## 4. Retired Alternatives

These options were discussed during redesign but are not part of the current
benchmark suite.

### 4.1 Retired from L1

- `H2 (equilibrium)`: too easy to separate methods meaningfully.
- `BH`: too hard in the current benchmark representation for a clean L1 role.

### 4.2 Retired from L2

- `BF`: mixes asymmetry with stronger higher-order and mixed-structure effects.
- `BeH+`: introduces charged-system interpretation and additional degeneracy
  concerns.

### 4.3 Retired from L3

- `HeH+`: charged-system story is less clean.
- `LiH Stretch`: more mixed and more dominated by bond-stretch effects.
- `H3+ (triangle)`: charged-system interpretation is less clean for the main
  benchmark.

### 4.4 Retired from L4 as main anchor

- `H3_Linear`: interesting, but mixes in stronger degeneracy / sector concerns.
- `H2O` alone: useful, but too heavy to define L4 by itself.
- `N2` / `C2` at 6 qubits: would require truncation that is too aggressive for
  the current benchmark story.

### 4.5 Retired from L5

- `H3 Linear`: less clean than `H4 Chain` for a dedicated topology benchmark.

### 4.6 Retired from L6 main commitment

- legacy single-case `L6_BeH2_Scalability_10q`: superseded by the ladder view.
- `BeH2 14q` as mandatory main rung: retained only as an optional extension.

---

## 5. Usage Notes For Paper Writing

When this benchmark is described in the paper, the following points should stay
consistent.

1. Call the dataset a `6-level active-space molecular benchmark` under
   `Jordan-Wigner` mapping.
2. State clearly that the main benchmark uses the unrestricted full-space qubit
   Hamiltonian.
3. Treat charge- or spin-restricted sector analysis as supplementary diagnosis.
4. Present L4 as a representation / entanglement-burden level rather than a
   k-locality level.
5. Present L6 as a same-family scalability ladder, with `14q` positioned as an
   optional extension rather than a required core rung.
