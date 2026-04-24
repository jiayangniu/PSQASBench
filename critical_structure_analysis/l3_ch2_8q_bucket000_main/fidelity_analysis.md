# Fidelity Analysis

- circuits analyzed: `10`
- fidelity threshold: `0.999` (deterministic complete-linkage)
- optimizer: `inherit`  |  n_restarts: `20`
- cobyla_maxiter fallback: `2000`  |  rotosolve_sweeps fallback: `2`
- distinct re-optimized state clusters: **2**

> **What this measures**: states produced by re-optimizing each retained
> circuit structure independently from scratch.  n_clusters == 1 means
> all CNOT skeletons fall into one high-fidelity state family at the chosen
> threshold after re-optimization.  It does NOT prove identical variational
> basins in the original training, nor exact physical identity beyond that
> threshold.

## Per-Circuit Results

| # | Label | Gates | Rot | Optimizer | Params | Error (mHa) | Cluster |
|--:|-------|------:|----:|:----------|:-------|------------:|:-------:|
| 0 | `circuit_0` | 4 | 2 | rotosolve | sweeps=2 | 0.0000 | 0 |
| 1 | `circuit_1` | 10 | 8 | rotosolve | sweeps=2 | 0.0000 | 0 |
| 2 | `circuit_2` | 8 | 3 | rotosolve | sweeps=2 | 0.0000 | 0 |
| 3 | `circuit_3` | 9 | 3 | rotosolve | sweeps=2 | 0.0000 | 0 |
| 4 | `circuit_4` | 5 | 3 | rotosolve | sweeps=2 | 0.0000 | 0 |
| 5 | `circuit_5` | 4 | 2 | rotosolve | sweeps=2 | 0.0000 | 1 |
| 6 | `circuit_6` | 10 | 4 | rotosolve | sweeps=2 | 0.0000 | 0 |
| 7 | `circuit_7` | 4 | 3 | rotosolve | sweeps=2 | 0.0000 | 1 |
| 8 | `circuit_8` | 17 | 6 | rotosolve | sweeps=2 | 0.0000 | 1 |
| 9 | `circuit_9` | 6 | 3 | rotosolve | sweeps=2 | 0.0000 | 0 |

## Pairwise Fidelity Matrix

Values: `|⟨ψᵢ|ψⱼ⟩|²`.  Off-diagonal values ≥ threshold are **bold**.

| | `circuit_0` | `circuit_1` | `circuit_2` | `circuit_3` | `circuit_4` | `circuit_5` | `circuit_6` | `circuit_7` | `circuit_8` | `circuit_9` |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `circuit_0` | 1.0000 | **1.0000** | **1.0000** | **1.0000** | **1.0000** | 0.0000 | **1.0000** | 0.0000 | 0.0000 | **1.0000** |
| `circuit_1` | **1.0000** | 1.0000 | **1.0000** | **1.0000** | **1.0000** | 0.0000 | **1.0000** | 0.0000 | 0.0000 | **1.0000** |
| `circuit_2` | **1.0000** | **1.0000** | 1.0000 | **1.0000** | **1.0000** | 0.0000 | **1.0000** | 0.0000 | 0.0000 | **1.0000** |
| `circuit_3` | **1.0000** | **1.0000** | **1.0000** | 1.0000 | **1.0000** | 0.0000 | **1.0000** | 0.0000 | 0.0000 | **1.0000** |
| `circuit_4` | **1.0000** | **1.0000** | **1.0000** | **1.0000** | 1.0000 | 0.0000 | **1.0000** | 0.0000 | 0.0000 | **1.0000** |
| `circuit_5` | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0000 | **1.0000** | **1.0000** | 0.0000 |
| `circuit_6` | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | 0.0000 | 1.0000 | 0.0000 | 0.0000 | **1.0000** |
| `circuit_7` | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | **1.0000** | 0.0000 | 1.0000 | **1.0000** | 0.0000 |
| `circuit_8` | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | **1.0000** | 0.0000 | **1.0000** | 1.0000 | 0.0000 |
| `circuit_9` | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | 0.0000 | **1.0000** | 0.0000 | 0.0000 | 1.0000 |

## Cluster Summary

Deterministic complete-linkage clusters at fidelity ≥ 0.999: **2**

**Cluster 0** (7 circuit(s), mean re-opt err 0.0000 mHa): `circuit_0`, `circuit_1`, `circuit_2`, `circuit_3`, `circuit_4`, `circuit_6`, `circuit_9`
**Cluster 1** (3 circuit(s), mean re-opt err 0.0000 mHa): `circuit_5`, `circuit_7`, `circuit_8`

> **Finding**: 2 distinct re-optimized state cluster(s)
> detected (minimum cross-cluster fidelity = 0.0000).
> Possible explanations:
> (a) Genuine near-degeneracy — multiple variational solutions at the
>     same energy level (check cluster energy errors for equality).
> (b) Expressibility gap — some CNOT skeletons cannot reach the ground
>     state at all (check whether high-error circuits form their own cluster).
> (c) Insufficient restarts — increase --n-restarts to rule out (c) first.
