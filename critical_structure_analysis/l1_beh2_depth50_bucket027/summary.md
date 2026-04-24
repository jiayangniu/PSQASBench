# Summary

- target error threshold: `saved snapshot events / legacy accept_err fallback`
- selected bucket: `0.27 mHa`
- bucket width: `0.01 mHa`
- discovered runs: `1`
- snapshot events in bucket: `5`
- selected snapshot events for pruning: `4`

**Anchor Actions**

- `RZ(q=5)`: 1
- `CNOT(4->0)`: 1
- `CNOT(5->0)`: 1

**Per-Snapshot Pruning Summary**

| Snapshot | Baseline (mHa) | Retained (mHa) | Δerror (mHa) | Original | Fixed | Max Steps | Retained | Removed | Redundancy | Optimizer |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `crlqas__L1_BeH2_STO3G_6q__L1_BeH2_STO3G_6q_cobyla_20k_depth50__seed11111__ep699__snap0` | 0.268228 | 0.268254 | 0.000026 | 18 | 0 | 18 | 8 | 10 | 55.56% | cobyla |
| `crlqas__L1_BeH2_STO3G_6q__L1_BeH2_STO3G_6q_cobyla_20k_depth50__seed11111__ep295__snap0` | 0.268621 | 0.268225 | -0.000396 | 29 | 3 | 26 | 9 | 20 | 68.97% | cobyla |
| `crlqas__L1_BeH2_STO3G_6q__L1_BeH2_STO3G_6q_cobyla_20k_depth50__seed11111__ep456__snap0` | 0.268226 | 0.268225 | -0.000001 | 38 | 4 | 34 | 14 | 24 | 63.16% | cobyla |
| `crlqas__L1_BeH2_STO3G_6q__L1_BeH2_STO3G_6q_cobyla_20k_depth50__seed11111__ep582__snap0` | 0.268226 | 0.268225 | -0.000000 | 9 | 0 | 9 | 6 | 3 | 33.33% | cobyla |

**Exact Retained-Structure Matches**

- count=1: `RX(q=1,θ=+0.039) | CNOT(1->5) | CNOT(1->0) | RZ(q=0,θ=-1.561) | CNOT(1->2) | RX(q=5,θ=-3.142) | CNOT(0->2) | CNOT(5->4)`
- count=1: `RY(q=1,θ=+3.142) | RY(q=4,θ=-3.180) | RY(q=0,θ=+3.142) | RY(q=2,θ=+0.000) | CNOT(4->0) | CNOT(4->2) | CNOT(2->1) | CNOT(4->5) | CNOT(5->2)`
- count=1: `RY(q=2,θ=+0.039) | RY(q=3,θ=-3.142) | RY(q=5,θ=-3.142) | CNOT(2->1) | CNOT(5->4) | CNOT(5->1) | CNOT(1->3) | CNOT(3->5) | CNOT(3->4) | CNOT(3->4) | CNOT(5->0) | CNOT(0->1) | CNOT(3->4) | CNOT(5->0)`
- count=1: `RY(q=1,θ=+0.039) | CNOT(1->4) | CNOT(1->0) | RX(q=5,θ=-3.142) | RX(q=4,θ=+3.142) | CNOT(0->5)`

**Common Retained Gate Signatures**

- none
