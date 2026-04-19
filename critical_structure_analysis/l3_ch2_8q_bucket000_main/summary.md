# Summary

- target error threshold: `inherit_from_run_meta`
- selected bucket: `0.00 mHa`
- discovered runs: `1`
- hit episodes in bucket: `77`
- selected episodes for pruning: `10`

**Anchor Actions**

- `CNOT(0->4)`: 13
- `CNOT(1->5)`: 7
- `CNOT(4->0)`: 5

**Per-Episode Pruning Summary**

| Episode | Baseline (mHa) | Retained (mHa) | Δerror (mHa) | Original | Fixed | Max Steps | Retained | Removed | Redundancy | Optimizer | Retained Gates |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| `crlqas__L3_CH2_Singlet_8q__L3_CH2_Singlet_8q_rotosolve_s2_check__seed11111__ep1973` | 0.000000 | 0.000000 | 0.000000 | 9 | 1 | 8 | 4 | 5 | 55.56% | rotosolve | `RY(q=0) | RY(q=6) | CNOT(6->4) | CNOT(0->4)` |
| `crlqas__L3_CH2_Singlet_8q__L3_CH2_Singlet_8q_rotosolve_s2_check__seed11111__ep1774` | 0.000000 | 0.000000 | 0.000000 | 23 | 2 | 21 | 10 | 13 | 56.52% | rotosolve | `RY(q=0) | RX(q=6) | RX(q=4) | CNOT(6->7) | CNOT(0->4) | RX(q=7) | RX(q=7) | RX(q=7) | RX(q=7) | RX(q=7)` |
| `crlqas__L3_CH2_Singlet_8q__L3_CH2_Singlet_8q_rotosolve_s2_check__seed11111__ep1359` | 0.000000 | 0.000000 | 0.000000 | 64 | 1 | 63 | 8 | 56 | 87.50% | rotosolve | `RX(q=6) | CNOT(6->5) | CNOT(5->1) | RX(q=5) | CNOT(6->1) | CNOT(6->4) | RY(q=0) | CNOT(0->4)` |
| `crlqas__L3_CH2_Singlet_8q__L3_CH2_Singlet_8q_rotosolve_s2_check__seed11111__ep1606` | 0.000000 | 0.000000 | 0.000000 | 50 | 2 | 48 | 9 | 41 | 82.00% | rotosolve | `RY(q=5) | CNOT(5->4) | CNOT(4->5) | RX(q=7) | CNOT(4->1) | RX(q=6) | CNOT(1->0) | CNOT(4->1) | CNOT(6->0)` |
| `crlqas__L3_CH2_Singlet_8q__L3_CH2_Singlet_8q_rotosolve_s2_check__seed11111__ep1920` | 0.000000 | 0.000000 | 0.000000 | 57 | 3 | 54 | 5 | 52 | 91.23% | rotosolve | `CNOT(6->4) | RX(q=0) | RY(q=4) | CNOT(4->0) | RY(q=6)` |
| `crlqas__L3_CH2_Singlet_8q__L3_CH2_Singlet_8q_rotosolve_s2_check__seed11111__ep1406` | 0.000000 | 0.000000 | 0.000000 | 68 | 3 | 65 | 4 | 64 | 94.12% | rotosolve | `RY(q=1) | RY(q=5) | CNOT(1->7) | CNOT(5->1)` |
| `crlqas__L3_CH2_Singlet_8q__L3_CH2_Singlet_8q_rotosolve_s2_check__seed11111__ep1913` | 0.000000 | 0.000000 | 0.000000 | 49 | 3 | 46 | 10 | 39 | 79.59% | rotosolve | `RY(q=3) | RY(q=4) | CNOT(3->6) | CNOT(4->7) | RX(q=3) | RX(q=4) | CNOT(7->2) | CNOT(7->0) | CNOT(7->2) | CNOT(0->7)` |
| `crlqas__L3_CH2_Singlet_8q__L3_CH2_Singlet_8q_rotosolve_s2_check__seed11111__ep1789` | 0.000000 | 0.000000 | 0.000000 | 25 | 0 | 25 | 4 | 21 | 84.00% | rotosolve | `RX(q=7) | RY(q=5) | CNOT(5->1) | RY(q=5)` |
| `crlqas__L3_CH2_Singlet_8q__L3_CH2_Singlet_8q_rotosolve_s2_check__seed11111__ep1646` | 0.000000 | 0.000000 | 0.000000 | 58 | 2 | 56 | 17 | 41 | 70.69% | rotosolve | `RY(q=7) | RX(q=6) | CNOT(6->2) | CNOT(2->3) | CNOT(3->6) | CNOT(7->0) | RY(q=2) | CNOT(3->4) | RY(q=0) | CNOT(3->5) | CNOT(4->1) | CNOT(7->3) | RY(q=1) | CNOT(5->4) | CNOT(7->3) | RY(q=3) | CNOT(1->5)` |
| `crlqas__L3_CH2_Singlet_8q__L3_CH2_Singlet_8q_rotosolve_s2_check__seed11111__ep1341` | 0.000000 | 0.000000 | 0.000000 | 37 | 2 | 35 | 6 | 31 | 83.78% | rotosolve | `RY(q=1) | CNOT(1->6) | RY(q=1) | RY(q=4) | CNOT(6->0) | CNOT(4->0)` |

**Exact Retained-Structure Matches**

- count=1: `RY(q=0) | RY(q=6) | CNOT(6->4) | CNOT(0->4)`
- count=1: `RY(q=0) | RX(q=6) | RX(q=4) | CNOT(6->7) | CNOT(0->4) | RX(q=7) | RX(q=7) | RX(q=7) | RX(q=7) | RX(q=7)`
- count=1: `RX(q=6) | CNOT(6->5) | CNOT(5->1) | RX(q=5) | CNOT(6->1) | CNOT(6->4) | RY(q=0) | CNOT(0->4)`
- count=1: `RY(q=5) | CNOT(5->4) | CNOT(4->5) | RX(q=7) | CNOT(4->1) | RX(q=6) | CNOT(1->0) | CNOT(4->1) | CNOT(6->0)`
- count=1: `CNOT(6->4) | RX(q=0) | RY(q=4) | CNOT(4->0) | RY(q=6)`
- count=1: `RY(q=1) | RY(q=5) | CNOT(1->7) | CNOT(5->1)`
- count=1: `RY(q=3) | RY(q=4) | CNOT(3->6) | CNOT(4->7) | RX(q=3) | RX(q=4) | CNOT(7->2) | CNOT(7->0) | CNOT(7->2) | CNOT(0->7)`
- count=1: `RX(q=7) | RY(q=5) | CNOT(5->1) | RY(q=5)`
- count=1: `RY(q=7) | RX(q=6) | CNOT(6->2) | CNOT(2->3) | CNOT(3->6) | CNOT(7->0) | RY(q=2) | CNOT(3->4) | RY(q=0) | CNOT(3->5) | CNOT(4->1) | CNOT(7->3) | RY(q=1) | CNOT(5->4) | CNOT(7->3) | RY(q=3) | CNOT(1->5)`
- count=1: `RY(q=1) | CNOT(1->6) | RY(q=1) | RY(q=4) | CNOT(6->0) | CNOT(4->0)`

**Common Retained Gate Signatures**

- none
