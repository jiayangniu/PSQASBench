# Fidelity Analysis

- circuits analyzed: `10`
- fidelity threshold: `0.999` (deterministic complete-linkage)
- optimizer: `inherit`  |  n_restarts: `20`
- cobyla_maxiter fallback: `2000`  |  rotosolve_sweeps fallback: `2`
- distinct re-optimized state clusters: **4**

> **What this measures**: states produced by re-optimizing each retained
> circuit structure independently from scratch.  n_clusters == 1 means
> all CNOT skeletons fall into one high-fidelity state family at the chosen
> threshold after re-optimization.  It does NOT prove identical variational
> basins in the original training, nor exact physical identity beyond that
> threshold.

## Per-Circuit Results

| # | Label | Gates | Rot | Optimizer | Params | Error (mHa) | Cluster |
|--:|-------|------:|----:|:----------|:-------|------------:|:-------:|
| 0 | `ep568_s0` | 17 | 5 | cobyla | maxiter=100 | 0.2415 | 0 |
| 1 | `ep790_s0` | 17 | 7 | cobyla | maxiter=100 | 0.6666 | 2 |
| 2 | `ep10240_s0` | 16 | 6 | cobyla | maxiter=100 | 0.2479 | 0 |
| 3 | `ep226_s0` | 24 | 7 | cobyla | maxiter=100 | 0.2807 | 0 |
| 4 | `ep2171_s0` | 13 | 5 | cobyla | maxiter=100 | 0.2281 | 0 |
| 5 | `ep173_s0` | 17 | 7 | cobyla | maxiter=100 | 1.7264 | 1 |
| 6 | `ep339_s0` | 16 | 6 | cobyla | maxiter=100 | 0.3744 | 0 |
| 7 | `ep9776_s0` | 18 | 9 | cobyla | maxiter=100 | 4.9310 | 3 |
| 8 | `ep19396_s0` | 10 | 5 | cobyla | maxiter=100 | 0.2292 | 0 |
| 9 | `ep1333_s0` | 20 | 7 | cobyla | maxiter=100 | 0.2535 | 0 |

## Pairwise Fidelity Matrix

Values: `|⟨ψᵢ|ψⱼ⟩|²`.  Off-diagonal values ≥ threshold are **bold**.

| | `ep568_s0` | `ep790_s0` | `ep10240_s0` | `ep226_s0` | `ep2171_s0` | `ep173_s0` | `ep339_s0` | `ep9776_s0` | `ep19396_s0` | `ep1333_s0` |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `ep568_s0` | 1.0000 | 0.9980 | **1.0000** | **0.9997** | **1.0000** | 0.9824 | **0.9999** | 0.9487 | **1.0000** | **1.0000** |
| `ep790_s0` | 0.9980 | 1.0000 | 0.9980 | 0.9985 | 0.9979 | 0.9806 | 0.9982 | 0.9471 | 0.9981 | 0.9980 |
| `ep10240_s0` | **1.0000** | 0.9980 | 1.0000 | **0.9996** | **0.9999** | 0.9824 | **0.9999** | 0.9487 | **0.9999** | **0.9999** |
| `ep226_s0` | **0.9997** | 0.9985 | **0.9996** | 1.0000 | **0.9998** | 0.9819 | **0.9994** | 0.9488 | **0.9999** | **0.9996** |
| `ep2171_s0` | **1.0000** | 0.9979 | **0.9999** | **0.9998** | 1.0000 | 0.9823 | **0.9998** | 0.9488 | **1.0000** | **0.9999** |
| `ep173_s0` | 0.9824 | 0.9806 | 0.9824 | 0.9819 | 0.9823 | 1.0000 | 0.9824 | 0.9704 | 0.9823 | 0.9824 |
| `ep339_s0` | **0.9999** | 0.9982 | **0.9999** | **0.9994** | **0.9998** | 0.9824 | 1.0000 | 0.9487 | **0.9998** | **0.9999** |
| `ep9776_s0` | 0.9487 | 0.9471 | 0.9487 | 0.9488 | 0.9488 | 0.9704 | 0.9487 | 1.0000 | 0.9488 | 0.9487 |
| `ep19396_s0` | **1.0000** | 0.9981 | **0.9999** | **0.9999** | **1.0000** | 0.9823 | **0.9998** | 0.9488 | 1.0000 | **0.9999** |
| `ep1333_s0` | **1.0000** | 0.9980 | **0.9999** | **0.9996** | **0.9999** | 0.9824 | **0.9999** | 0.9487 | **0.9999** | 1.0000 |

## Cluster Summary

Deterministic complete-linkage clusters at fidelity ≥ 0.999: **4**

**Cluster 0** (7 circuit(s), mean re-opt err 0.2650 mHa): `ep568_s0`, `ep10240_s0`, `ep226_s0`, `ep2171_s0`, `ep339_s0`, `ep19396_s0`, `ep1333_s0`
**Cluster 1** (1 circuit(s), mean re-opt err 1.7264 mHa): `ep173_s0`
**Cluster 2** (1 circuit(s), mean re-opt err 0.6666 mHa): `ep790_s0`
**Cluster 3** (1 circuit(s), mean re-opt err 4.9310 mHa): `ep9776_s0`

> **Finding**: 4 distinct re-optimized state cluster(s)
> detected (minimum cross-cluster fidelity = 0.9471).
> Possible explanations:
> (a) Genuine near-degeneracy — multiple variational solutions at the
>     same energy level (check cluster energy errors for equality).
> (b) Expressibility gap — some CNOT skeletons cannot reach the ground
>     state at all (check whether high-error circuits form their own cluster).
> (c) Insufficient restarts — increase --n-restarts to rule out (c) first.
