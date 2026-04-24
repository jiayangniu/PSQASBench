# Summary

- target error threshold: `saved snapshot events / legacy accept_err fallback`
- selected bucket: `2.0 mHa`
- bucket width: `0.5 mHa`
- discovered runs: `1`
- snapshot events in bucket: `4`
- selected snapshot events for pruning: `4`

**Anchor Actions**

- `CNOT(1->2)`: 2
- `CNOT(3->5)`: 2

**Per-Snapshot Pruning Summary**

| Snapshot | Baseline (mHa) | Retained (mHa) | Δerror (mHa) | Original | Fixed | Max Steps | Retained | Removed | Redundancy | Optimizer | Retained Gates |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| `crlqas__L5_H3_Linear_6q__L5_H3_Linear_6q_cobyla_20k_all__seed11111__ep8901__snap1` | 1.337114 | 1.320397 | -0.016718 | 47 | 9 | 38 | 33 | 14 | 29.79% | cobyla | `RX(q=0) | RX(q=3) | CNOT(3->5) | CNOT(3->4) | CNOT(0->1) | CNOT(0->2) | CNOT(0->5) | CNOT(1->2) | CNOT(1->4) | CNOT(1->4) | CNOT(1->4) | CNOT(1->2) | CNOT(1->2) | RY(q=3) | CNOT(3->0) | CNOT(3->0) | CNOT(3->0) | CNOT(3->0) | CNOT(3->0) | RX(q=0) | CNOT(5->2) | CNOT(5->2) | CNOT(5->2) | CNOT(5->2) | CNOT(0->4) | CNOT(5->2) | RZ(q=0) | CNOT(1->4) | CNOT(0->1) | CNOT(0->1) | CNOT(1->2) | CNOT(1->2) | CNOT(1->2)` |
| `crlqas__L5_H3_Linear_6q__L5_H3_Linear_6q_cobyla_20k_all__seed11111__ep10560__snap0` | 1.404599 | 1.460085 | 0.055486 | 43 | 2 | 41 | 29 | 14 | 32.56% | cobyla | `RX(q=0) | RX(q=2) | RX(q=5) | CNOT(5->2) | CNOT(5->2) | CNOT(2->5) | RY(q=1) | CNOT(2->0) | CNOT(0->3) | RX(q=4) | CNOT(3->0) | CNOT(3->0) | CNOT(3->0) | CNOT(4->1) | CNOT(3->0) | CNOT(0->5) | CNOT(2->1) | CNOT(3->4) | CNOT(5->3) | CNOT(3->0) | CNOT(1->0) | RX(q=5) | CNOT(2->1) | CNOT(2->0) | CNOT(0->1) | RZ(q=4) | CNOT(3->4) | CNOT(3->0) | CNOT(3->5)` |
| `crlqas__L5_H3_Linear_6q__L5_H3_Linear_6q_cobyla_20k_all__seed11111__ep10560__snap0` | 1.404599 | 1.460085 | 0.055486 | 43 | 2 | 41 | 29 | 14 | 32.56% | cobyla | `RX(q=0) | RX(q=2) | RX(q=5) | CNOT(5->2) | CNOT(5->2) | CNOT(2->5) | RY(q=1) | CNOT(2->0) | CNOT(0->3) | RX(q=4) | CNOT(3->0) | CNOT(3->0) | CNOT(3->0) | CNOT(4->1) | CNOT(3->0) | CNOT(0->5) | CNOT(2->1) | CNOT(3->4) | CNOT(5->3) | CNOT(3->0) | CNOT(1->0) | RX(q=5) | CNOT(2->1) | CNOT(2->0) | CNOT(0->1) | RZ(q=4) | CNOT(3->4) | CNOT(3->0) | CNOT(3->5)` |
| `crlqas__L5_H3_Linear_6q__L5_H3_Linear_6q_cobyla_20k_all__seed11111__ep8901__snap1` | 1.337114 | 1.320397 | -0.016718 | 47 | 9 | 38 | 33 | 14 | 29.79% | cobyla | `RX(q=0) | RX(q=3) | CNOT(3->5) | CNOT(3->4) | CNOT(0->1) | CNOT(0->2) | CNOT(0->5) | CNOT(1->2) | CNOT(1->4) | CNOT(1->4) | CNOT(1->4) | CNOT(1->2) | CNOT(1->2) | RY(q=3) | CNOT(3->0) | CNOT(3->0) | CNOT(3->0) | CNOT(3->0) | CNOT(3->0) | RX(q=0) | CNOT(5->2) | CNOT(5->2) | CNOT(5->2) | CNOT(5->2) | CNOT(0->4) | CNOT(5->2) | RZ(q=0) | CNOT(1->4) | CNOT(0->1) | CNOT(0->1) | CNOT(1->2) | CNOT(1->2) | CNOT(1->2)` |

**Exact Retained-Structure Matches**

- count=2: `RX(q=0) | RX(q=3) | CNOT(3->5) | CNOT(3->4) | CNOT(0->1) | CNOT(0->2) | CNOT(0->5) | CNOT(1->2) | CNOT(1->4) | CNOT(1->4) | CNOT(1->4) | CNOT(1->2) | CNOT(1->2) | RY(q=3) | CNOT(3->0) | CNOT(3->0) | CNOT(3->0) | CNOT(3->0) | CNOT(3->0) | RX(q=0) | CNOT(5->2) | CNOT(5->2) | CNOT(5->2) | CNOT(5->2) | CNOT(0->4) | CNOT(5->2) | RZ(q=0) | CNOT(1->4) | CNOT(0->1) | CNOT(0->1) | CNOT(1->2) | CNOT(1->2) | CNOT(1->2)`
- count=2: `RX(q=0) | RX(q=2) | RX(q=5) | CNOT(5->2) | CNOT(5->2) | CNOT(2->5) | RY(q=1) | CNOT(2->0) | CNOT(0->3) | RX(q=4) | CNOT(3->0) | CNOT(3->0) | CNOT(3->0) | CNOT(4->1) | CNOT(3->0) | CNOT(0->5) | CNOT(2->1) | CNOT(3->4) | CNOT(5->3) | CNOT(3->0) | CNOT(1->0) | RX(q=5) | CNOT(2->1) | CNOT(2->0) | CNOT(0->1) | RZ(q=4) | CNOT(3->4) | CNOT(3->0) | CNOT(3->5)`

**Common Retained Gate Signatures**

- `CNOT(0->1)`
- `CNOT(0->5)`
- `CNOT(3->0)`
- `CNOT(3->4)`
- `CNOT(3->5)`
- `CNOT(5->2)`
- `RX(q=0)`
