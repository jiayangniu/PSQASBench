# Summary

- target error threshold: `1.6000 mHa`
- selected bucket: `0.26 mHa`
- discovered runs: `1`
- hit episodes in bucket: `1`
- selected episodes for pruning: `1`

**Anchor Actions**

- `RY(q=1)`: 1

**Per-Episode Pruning Summary**

| Episode | Baseline (mHa) | Retained (mHa) | Δerror (mHa) | Original | Fixed | Max Steps | Retained | Removed | Redundancy | Optimizer | Retained Gates |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| `crlqas__L2_LiH_Equil_6q__L2_LiH_Equil_6q_cobyla_20k__seed11111__ep1462` | 0.228133 | 0.228115 | -0.000018 | 17 | 1 | 16 | 8 | 9 | 52.94% | cobyla | `RY(q=0) | CNOT(0->4) | CNOT(4->1) | CNOT(1->5) | CNOT(5->4) | RY(q=0) | CNOT(1->4) | RY(q=1)` |

**Exact Retained-Structure Matches**

- count=1: `RY(q=0) | CNOT(0->4) | CNOT(4->1) | CNOT(1->5) | CNOT(5->4) | RY(q=0) | CNOT(1->4) | RY(q=1)`

**Common Retained Gate Signatures**

- `CNOT(0->4)`
- `CNOT(1->4)`
- `CNOT(1->5)`
- `CNOT(4->1)`
- `CNOT(5->4)`
- `RY(q=0)`
- `RY(q=1)`
