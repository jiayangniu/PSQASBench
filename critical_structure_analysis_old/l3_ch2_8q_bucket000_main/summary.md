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
| `crlqas__L3_CH2_Singlet_8q__L3_CH2_Singlet_8q_rotosolve_s2_check__seed11111__ep1973` | 5.662392 | 5.662392 | 0.000000 | 9 | 1 | 8 | 2 | 7 | 77.78% | rotosolve | `RY(q=6) | CNOT(6->4)` |
| `crlqas__L3_CH2_Singlet_8q__L3_CH2_Singlet_8q_rotosolve_s2_check__seed11111__ep1774` | 240.503060 | 240.503060 | 0.000000 | 23 | 2 | 21 | 2 | 21 | 91.30% | rotosolve | `RX(q=6) | CNOT(6->7)` |
| `crlqas__L3_CH2_Singlet_8q__L3_CH2_Singlet_8q_rotosolve_s2_check__seed11111__ep1359` | 0.000000 | 0.000000 | 0.000000 | 64 | 1 | 63 | 8 | 56 | 87.50% | rotosolve | `RX(q=6) | CNOT(6->5) | CNOT(5->1) | RX(q=5) | CNOT(6->1) | CNOT(6->4) | RY(q=0) | CNOT(0->4)` |
| `crlqas__L3_CH2_Singlet_8q__L3_CH2_Singlet_8q_rotosolve_s2_check__seed11111__ep1606` | 5.662392 | 5.662392 | 0.000000 | 50 | 2 | 48 | 5 | 45 | 90.00% | rotosolve | `RX(q=7) | CNOT(7->3) | CNOT(7->3) | RY(q=5) | CNOT(6->0)` |
| `crlqas__L3_CH2_Singlet_8q__L3_CH2_Singlet_8q_rotosolve_s2_check__seed11111__ep1920` | 450.380646 | 450.380646 | 0.000000 | 57 | 3 | 54 | 3 | 54 | 94.74% | rotosolve | `CNOT(4->0) | CNOT(6->2) | RY(q=6)` |
| `crlqas__L3_CH2_Singlet_8q__L3_CH2_Singlet_8q_rotosolve_s2_check__seed11111__ep1406` | 571.618406 | 571.618406 | 0.000000 | 68 | 3 | 65 | 32 | 36 | 52.94% | rotosolve | `RX(q=0) | CNOT(0->4) | CNOT(4->3) | CNOT(3->2) | CNOT(4->5) | CNOT(5->0) | CNOT(3->6) | CNOT(3->1) | CNOT(2->0) | RY(q=1) | CNOT(2->5) | CNOT(4->1) | CNOT(3->0) | CNOT(4->1) | CNOT(4->1) | CNOT(3->7) | CNOT(4->1) | CNOT(3->7) | CNOT(3->5) | CNOT(6->1) | CNOT(2->7) | CNOT(5->7) | CNOT(6->0) | CNOT(1->7) | CNOT(2->7) | CNOT(3->6) | CNOT(3->2) | CNOT(3->0) | CNOT(3->0) | CNOT(5->1) | CNOT(3->5) | CNOT(3->0)` |
| `crlqas__L3_CH2_Singlet_8q__L3_CH2_Singlet_8q_rotosolve_s2_check__seed11111__ep1913` | 0.000000 | 0.000000 | 0.000000 | 49 | 3 | 46 | 10 | 39 | 79.59% | rotosolve | `RY(q=3) | RY(q=4) | CNOT(3->6) | CNOT(4->7) | RX(q=3) | RX(q=4) | CNOT(7->2) | CNOT(7->0) | CNOT(7->2) | CNOT(0->7)` |
| `crlqas__L3_CH2_Singlet_8q__L3_CH2_Singlet_8q_rotosolve_s2_check__seed11111__ep1789` | 696.943502 | 696.943502 | 0.000000 | 25 | 0 | 25 | 3 | 22 | 88.00% | rotosolve | `RY(q=4) | CNOT(4->2) | CNOT(2->7)` |
| `crlqas__L3_CH2_Singlet_8q__L3_CH2_Singlet_8q_rotosolve_s2_check__seed11111__ep1646` | 378.597706 | 378.597706 | 0.000000 | 58 | 2 | 56 | 2 | 56 | 96.55% | rotosolve | `RY(q=4) | RY(q=1)` |
| `crlqas__L3_CH2_Singlet_8q__L3_CH2_Singlet_8q_rotosolve_s2_check__seed11111__ep1341` | 565.408647 | 565.408647 | 0.000000 | 37 | 2 | 35 | 2 | 35 | 94.59% | rotosolve | `RY(q=4) | CNOT(4->0)` |

**Exact Retained-Structure Matches**

- count=1: `RY(q=6) | CNOT(6->4)`
- count=1: `RX(q=6) | CNOT(6->7)`
- count=1: `RX(q=6) | CNOT(6->5) | CNOT(5->1) | RX(q=5) | CNOT(6->1) | CNOT(6->4) | RY(q=0) | CNOT(0->4)`
- count=1: `RX(q=7) | CNOT(7->3) | CNOT(7->3) | RY(q=5) | CNOT(6->0)`
- count=1: `CNOT(4->0) | CNOT(6->2) | RY(q=6)`
- count=1: `RX(q=0) | CNOT(0->4) | CNOT(4->3) | CNOT(3->2) | CNOT(4->5) | CNOT(5->0) | CNOT(3->6) | CNOT(3->1) | CNOT(2->0) | RY(q=1) | CNOT(2->5) | CNOT(4->1) | CNOT(3->0) | CNOT(4->1) | CNOT(4->1) | CNOT(3->7) | CNOT(4->1) | CNOT(3->7) | CNOT(3->5) | CNOT(6->1) | CNOT(2->7) | CNOT(5->7) | CNOT(6->0) | CNOT(1->7) | CNOT(2->7) | CNOT(3->6) | CNOT(3->2) | CNOT(3->0) | CNOT(3->0) | CNOT(5->1) | CNOT(3->5) | CNOT(3->0)`
- count=1: `RY(q=3) | RY(q=4) | CNOT(3->6) | CNOT(4->7) | RX(q=3) | RX(q=4) | CNOT(7->2) | CNOT(7->0) | CNOT(7->2) | CNOT(0->7)`
- count=1: `RY(q=4) | CNOT(4->2) | CNOT(2->7)`
- count=1: `RY(q=4) | RY(q=1)`
- count=1: `RY(q=4) | CNOT(4->0)`

**Common Retained Gate Signatures**

- none
