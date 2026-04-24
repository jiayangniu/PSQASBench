# Summary

- target error threshold: `saved snapshot events / legacy accept_err fallback`
- selected bucket: `0.55 mHa`
- bucket width: `0.01 mHa`
- discovered runs: `1`
- snapshot events in bucket: `2`
- selected snapshot events for pruning: `2`
- episode sampling: `late_fraction = 1.0`

**Anchor Actions**

- `CNOT(1->2)`: 1
- `RZ(q=1)`: 1

**Per-Snapshot Pruning Summary**

| Snapshot | Baseline (mHa) | Retained (mHa) | Δerror (mHa) | Original | Fixed | Max Steps | Retained | Removed | Redundancy | Optimizer |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `qdarts__L1_BeH2_STO3G_6q__L1_BeH2_STO3G_6q__seed11111__ep16__snap0` | 0.554444 | 0.554445 | 0.000001 | 16 | 1 | 15 | 4 | 12 | 75.00% | cobyla |
| `qdarts__L1_BeH2_STO3G_6q__L1_BeH2_STO3G_6q__seed11111__ep18__snap0` | 0.554443 | 0.554443 | 0.000000 | 18 | 2 | 16 | 4 | 14 | 77.78% | cobyla |

**Exact Retained-Structure Matches**

- count=1: `RY(q=2,θ=+3.142) | CNOT(2->4) | CNOT(4->5) | CNOT(5->2)`
- count=1: `RY(q=5,θ=-3.142) | CNOT(5->1) | CNOT(1->4) | RY(q=1,θ=-3.142)`

**Common Retained Gate Signatures**

- none
