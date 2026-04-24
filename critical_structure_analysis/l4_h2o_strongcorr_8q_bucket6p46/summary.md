# Summary

- target error threshold: `saved snapshot events / legacy accept_err fallback`
- selected bucket: `6.46 mHa`
- bucket width: `0.01 mHa`
- discovered runs: `1`
- snapshot events in bucket: `69`
- selected snapshot events for pruning: `10`
- episode sampling: `late_fraction = 1.0`

**Anchor Actions**

- `CNOT(3->2)`: 5
- `CNOT(5->3)`: 3
- `CNOT(4->2)`: 3

**Per-Snapshot Pruning Summary**

| Snapshot | Baseline (mHa) | Retained (mHa) | Δerror (mHa) | Original | Fixed | Max Steps | Retained | Removed | Redundancy | Optimizer |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `crlqas__L4_H2O_StrongCorr_8q__L4_H2O_StrongCorr_8q_rotosolve_s2_20k__seed11111__ep9150__snap0` | 6.460776 | 6.460776 | 0.000000 | 53 | 1 | 52 | 31 | 22 | 41.51% | rotosolve |
| `crlqas__L4_H2O_StrongCorr_8q__L4_H2O_StrongCorr_8q_rotosolve_s2_20k__seed11111__ep17081__snap0` | 6.460776 | 6.460776 | 0.000000 | 28 | 2 | 26 | 16 | 12 | 42.86% | rotosolve |
| `crlqas__L4_H2O_StrongCorr_8q__L4_H2O_StrongCorr_8q_rotosolve_s2_20k__seed11111__ep5464__snap1` | 6.460776 | 6.460776 | 0.000000 | 17 | 0 | 17 | 13 | 4 | 23.53% | rotosolve |
| `crlqas__L4_H2O_StrongCorr_8q__L4_H2O_StrongCorr_8q_rotosolve_s2_20k__seed11111__ep8182__snap0` | 6.460776 | 6.460776 | 0.000000 | 35 | 0 | 35 | 17 | 18 | 51.43% | rotosolve |
| `crlqas__L4_H2O_StrongCorr_8q__L4_H2O_StrongCorr_8q_rotosolve_s2_20k__seed11111__ep19663__snap1` | 6.460776 | 6.460776 | 0.000000 | 41 | 1 | 40 | 25 | 16 | 39.02% | rotosolve |
| `crlqas__L4_H2O_StrongCorr_8q__L4_H2O_StrongCorr_8q_rotosolve_s2_20k__seed11111__ep19167__snap0` | 6.460776 | 6.460776 | 0.000000 | 45 | 0 | 45 | 26 | 19 | 42.22% | rotosolve |
| `crlqas__L4_H2O_StrongCorr_8q__L4_H2O_StrongCorr_8q_rotosolve_s2_20k__seed11111__ep4167__snap0` | 6.460776 | 6.460776 | 0.000000 | 50 | 1 | 49 | 27 | 23 | 46.00% | rotosolve |
| `crlqas__L4_H2O_StrongCorr_8q__L4_H2O_StrongCorr_8q_rotosolve_s2_20k__seed11111__ep2458__snap0` | 6.460776 | 6.460776 | 0.000000 | 48 | 2 | 46 | 26 | 22 | 45.83% | rotosolve |
| `crlqas__L4_H2O_StrongCorr_8q__L4_H2O_StrongCorr_8q_rotosolve_s2_20k__seed11111__ep8182__snap1` | 6.460776 | 6.460776 | 0.000000 | 38 | 0 | 38 | 19 | 19 | 50.00% | rotosolve |
| `crlqas__L4_H2O_StrongCorr_8q__L4_H2O_StrongCorr_8q_rotosolve_s2_20k__seed11111__ep3261__snap0` | 6.460776 | 6.460776 | -0.000000 | 64 | 2 | 62 | 43 | 21 | 32.81% | rotosolve |

**Exact Retained-Structure Matches**

- count=1: `RY(q=5,θ=-3.142) | RY(q=2,θ=+2.643) | CNOT(2->4) | CNOT(5->0) | CNOT(0->6) | CNOT(5->6) | CNOT(0->5) | CNOT(0->4) | CNOT(4->7) | RZ(q=5,θ=-3.142) | CNOT(7->3) | CNOT(6->7) | CNOT(6->7) | CNOT(6->7) | CNOT(2->5) | CNOT(2->5) | CNOT(4->7) | CNOT(2->5) | CNOT(7->2) | CNOT(1->0) | CNOT(0->6) | CNOT(7->4) | CNOT(6->7) | RY(q=0,θ=+3.142) | RY(q=0,θ=+0.000) | CNOT(7->4) | CNOT(7->4) | CNOT(0->6) | CNOT(0->6) | CNOT(7->4) | CNOT(6->2)`
- count=1: `RY(q=2,θ=+2.643) | CNOT(2->5) | RX(q=6,θ=-3.142) | CNOT(2->4) | CNOT(6->3) | RX(q=2,θ=-3.142) | CNOT(3->1) | CNOT(3->1) | CNOT(3->7) | CNOT(3->7) | CNOT(3->7) | CNOT(3->7) | CNOT(3->7) | CNOT(3->7) | CNOT(3->7) | CNOT(5->3)`
- count=1: `RY(q=5,θ=+2.643) | RX(q=7,θ=-3.142) | CNOT(5->2) | CNOT(7->4) | CNOT(4->6) | CNOT(7->2) | CNOT(4->1) | CNOT(1->4) | RY(q=1,θ=+3.142) | CNOT(2->3) | CNOT(5->4) | CNOT(5->1) | CNOT(5->1)`
- count=1: `RX(q=0,θ=+2.643) | CNOT(0->4) | CNOT(0->2) | CNOT(4->3) | CNOT(4->3) | CNOT(4->3) | CNOT(4->3) | CNOT(4->3) | CNOT(2->3) | RZ(q=4,θ=-1.571) | CNOT(0->7) | CNOT(7->5) | RX(q=2,θ=-3.142) | RY(q=6,θ=-3.142) | CNOT(2->3) | CNOT(2->7) | CNOT(4->0)`
- count=1: `RX(q=4,θ=+2.643) | RY(q=5,θ=-3.142) | CNOT(5->1) | CNOT(5->3) | RY(q=7,θ=-3.142) | CNOT(7->5) | CNOT(4->3) | CNOT(7->0) | CNOT(4->1) | CNOT(7->0) | CNOT(4->5) | CNOT(1->2) | CNOT(1->2) | CNOT(1->2) | CNOT(1->2) | CNOT(1->2) | RX(q=6,θ=-3.142) | CNOT(3->1) | CNOT(6->5) | RZ(q=4,θ=-1.571) | CNOT(6->5) | CNOT(6->2) | CNOT(6->2) | CNOT(6->2) | CNOT(6->2)`
- count=1: `RY(q=0,θ=-3.142) | RX(q=1,θ=-0.000) | RY(q=7,θ=-0.499) | CNOT(7->1) | CNOT(7->6) | CNOT(6->1) | CNOT(6->3) | CNOT(3->5) | RX(q=5,θ=-3.142) | CNOT(7->4) | CNOT(3->5) | CNOT(5->2) | CNOT(3->6) | CNOT(3->6) | RY(q=3,θ=-0.000) | CNOT(0->6) | RY(q=6,θ=-3.142) | CNOT(2->6) | CNOT(0->6) | CNOT(4->5) | CNOT(2->6) | CNOT(5->7) | CNOT(5->2) | CNOT(2->6) | CNOT(6->0) | CNOT(6->4)`
- count=1: `RY(q=0,θ=+2.643) | RY(q=4,θ=-3.142) | CNOT(0->5) | CNOT(5->0) | RX(q=6,θ=-3.142) | CNOT(4->7) | RX(q=1,θ=-3.142) | CNOT(1->2) | CNOT(4->3) | CNOT(4->3) | CNOT(4->3) | CNOT(3->4) | RZ(q=1,θ=-3.142) | RZ(q=7,θ=-3.142) | CNOT(3->4) | CNOT(3->1) | RZ(q=2,θ=-3.142) | RZ(q=2,θ=-3.142) | CNOT(1->3) | RZ(q=2,θ=-3.142) | CNOT(3->4) | CNOT(5->3) | CNOT(1->4) | CNOT(5->4) | CNOT(5->4) | CNOT(5->4) | CNOT(5->2)`
- count=1: `RY(q=5,θ=-0.499) | CNOT(5->3) | CNOT(5->6) | RY(q=6,θ=+3.142) | CNOT(5->0) | CNOT(5->0) | CNOT(5->0) | RY(q=7,θ=+3.142) | CNOT(6->1) | CNOT(0->2) | CNOT(3->7) | CNOT(7->1) | CNOT(5->1) | CNOT(7->1) | RX(q=6,θ=-3.142) | CNOT(5->0) | CNOT(7->1) | CNOT(7->6) | CNOT(7->1) | CNOT(3->4) | CNOT(1->5) | CNOT(2->1) | CNOT(0->3) | RX(q=4,θ=-3.142) | CNOT(4->1) | CNOT(2->7)`
- count=1: `RX(q=0,θ=+2.643) | CNOT(0->4) | CNOT(0->2) | CNOT(4->3) | CNOT(4->3) | CNOT(4->3) | CNOT(4->3) | CNOT(4->3) | CNOT(2->3) | RZ(q=4,θ=-1.571) | CNOT(0->7) | CNOT(7->5) | RX(q=2,θ=-3.142) | RY(q=6,θ=-3.142) | CNOT(2->3) | CNOT(2->7) | CNOT(4->0) | CNOT(6->7) | CNOT(6->7)`
- count=1: `RY(q=4,θ=-0.000) | RY(q=5,θ=-3.142) | CNOT(5->3) | CNOT(4->1) | CNOT(5->3) | CNOT(1->4) | RY(q=6,θ=+0.499) | CNOT(1->6) | CNOT(1->6) | CNOT(1->6) | CNOT(1->6) | CNOT(1->6) | CNOT(7->3) | CNOT(2->7) | RY(q=1,θ=+3.142) | CNOT(4->7) | CNOT(3->0) | CNOT(3->0) | CNOT(1->7) | CNOT(6->3) | RZ(q=1,θ=-3.142) | CNOT(6->3) | CNOT(6->2) | CNOT(7->3) | RZ(q=1,θ=-3.142) | RY(q=1,θ=+3.142) | CNOT(5->7) | CNOT(0->4) | CNOT(6->5) | CNOT(2->3) | CNOT(5->6) | CNOT(3->4) | CNOT(0->7) | CNOT(3->7) | CNOT(3->5) | CNOT(3->5) | RY(q=3,θ=-0.000) | CNOT(6->3) | RX(q=7,θ=-0.000) | RY(q=2,θ=-0.000) | CNOT(0->3) | RY(q=1,θ=+0.000) | CNOT(3->7)`

**Common Retained Gate Signatures**

- none
