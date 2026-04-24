# Summary

- target error threshold: `saved snapshot events / legacy accept_err fallback`
- selected bucket: `0.27 mHa`
- bucket width: `0.01 mHa`
- discovered runs: `1`
- snapshot events in bucket: `4`
- selected snapshot events for pruning: `4`

**Anchor Actions**

- `CNOT(5->4)`: 1
- `CNOT(0->4)`: 1
- `RY(q=5)`: 1

**Per-Snapshot Pruning Summary**

| Snapshot | Baseline (mHa) | Retained (mHa) | Δerror (mHa) | Original | Fixed | Max Steps | Retained | Removed | Redundancy | Optimizer |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `crlqas__L1_BeH2_STO3G_6q__L1_BeH2_STO3G_6q_cobyla_20k_depth10__seed11111__ep167__snap0` | 0.268225 | 0.268225 | -0.000000 | 9 | 1 | 8 | 6 | 3 | 33.33% | cobyla |
| `crlqas__L1_BeH2_STO3G_6q__L1_BeH2_STO3G_6q_cobyla_20k_depth10__seed11111__ep2816__snap0` | 0.268226 | 0.268225 | -0.000001 | 10 | 3 | 7 | 6 | 4 | 40.00% | cobyla |
| `crlqas__L1_BeH2_STO3G_6q__L1_BeH2_STO3G_6q_cobyla_20k_depth10__seed11111__ep6230__snap0` | 0.268225 | 0.268225 | 0.000000 | 8 | 1 | 7 | 5 | 3 | 37.50% | cobyla |
| `crlqas__L1_BeH2_STO3G_6q__L1_BeH2_STO3G_6q_cobyla_20k_depth10__seed11111__ep738__snap0` | 0.268228 | 0.268225 | -0.000003 | 9 | 2 | 7 | 6 | 3 | 33.33% | cobyla |

**Exact Retained-Structure Matches**

- count=1: `RY(q=2,θ=+3.103) | CNOT(2->5) | RY(q=3,θ=+3.142) | RX(q=2,θ=-3.142) | CNOT(5->3) | CNOT(5->4)`
- count=1: `RY(q=5,θ=-0.039) | CNOT(5->1) | CNOT(1->0) | RY(q=4,θ=+3.142) | CNOT(5->4) | RY(q=5,θ=+3.142)`
- count=1: `RY(q=4,θ=-0.039) | CNOT(4->2) | CNOT(2->3) | RY(q=4,θ=-3.142) | CNOT(4->5)`
- count=1: `RY(q=4,θ=-3.142) | RY(q=1,θ=-0.039) | CNOT(1->0) | CNOT(1->5) | RY(q=5,θ=-3.142) | CNOT(0->4)`

**Common Retained Gate Signatures**

- none
