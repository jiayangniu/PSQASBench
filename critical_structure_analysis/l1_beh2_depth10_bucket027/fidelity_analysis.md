# Fidelity Analysis

- circuits analyzed: `4`
- fidelity threshold: `0.999` (deterministic complete-linkage)
- optimizer: `inherit`  |  n_restarts: `20`
- cobyla_maxiter fallback: `2000`  |  rotosolve_sweeps fallback: `2`
- distinct re-optimized state clusters: **1**

> **What this measures**: states produced by re-optimizing each retained
> circuit structure independently from scratch.  n_clusters == 1 means
> all CNOT skeletons fall into one high-fidelity state family at the chosen
> threshold after re-optimization.  It does NOT prove identical variational
> basins in the original training, nor exact physical identity beyond that
> threshold.

## Per-Circuit Results

| # | Label | Gates | Rot | Optimizer | Params | Error (mHa) | Cluster |
|--:|-------|------:|----:|:----------|:-------|------------:|:-------:|
| 0 | `ep167_s0` | 6 | 3 | cobyla | maxiter=100 | 0.2682 | 0 |
| 1 | `ep2816_s0` | 6 | 3 | cobyla | maxiter=100 | 0.2682 | 0 |
| 2 | `ep6230_s0` | 5 | 2 | cobyla | maxiter=100 | 0.2682 | 0 |
| 3 | `ep738_s0` | 6 | 3 | cobyla | maxiter=100 | 0.2682 | 0 |

## Pairwise Fidelity Matrix

Values: `|⟨ψᵢ|ψⱼ⟩|²`.  Off-diagonal values ≥ threshold are **bold**.

| | `ep167_s0` | `ep2816_s0` | `ep6230_s0` | `ep738_s0` |
|:---|:---:|:---:|:---:|:---:|
| `ep167_s0` | 1.0000 | **0.9992** | **1.0000** | **0.9992** |
| `ep2816_s0` | **0.9992** | 1.0000 | **0.9992** | **1.0000** |
| `ep6230_s0` | **1.0000** | **0.9992** | 1.0000 | **0.9992** |
| `ep738_s0` | **0.9992** | **1.0000** | **0.9992** | 1.0000 |

## Cluster Summary

Deterministic complete-linkage clusters at fidelity ≥ 0.999: **1**

**Cluster 0** (4 circuit(s), mean re-opt err 0.2682 mHa): `ep167_s0`, `ep2816_s0`, `ep6230_s0`, `ep738_s0`

> **Finding**: When re-optimized from scratch, all retained structures
> fall into one fidelity cluster at threshold 0.999
> (minimum pairwise fidelity inside the cluster = 0.9992).
> Structural diversity in summary.md is therefore consistent with
> multiple retained circuits realizing the same re-optimized state
> family, but this alone does not prove identical original basins.
