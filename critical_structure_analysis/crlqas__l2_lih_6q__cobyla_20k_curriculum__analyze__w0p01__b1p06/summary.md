# Summary

- target error threshold: `saved snapshot events / legacy accept_err fallback`
- selected bucket: `1.06 mHa`
- bucket width: `0.01 mHa`
- discovered runs: `1`
- snapshot events in bucket: `1130`
- selected snapshot events for pruning: `10`

**Anchor Actions**

- `RX(q=4)`: 178
- `RX(q=5)`: 168
- `CNOT(4->5)`: 102

**Per-Snapshot Pruning Summary**

| Snapshot | Baseline (mHa) | Retained (mHa) | Δerror (mHa) | Original | Fixed | Max Steps | Retained | Removed | Redundancy | Optimizer | Retained Gates |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| `crlqas__L2_LiH_Equil_6q__L2_LiH_Equil_6q_cobyla_20k_curriculum__seed11111__ep17288__snap0` | 1.053897 | 1.053894 | -0.000003 | 4 | 1 | 3 | 2 | 2 | 50.00% | cobyla | `RY(q=4) | CNOT(4->5)` |
| `crlqas__L2_LiH_Equil_6q__L2_LiH_Equil_6q_cobyla_20k_curriculum__seed11111__ep9145__snap0` | 1.053896 | 1.053894 | -0.000002 | 19 | 3 | 16 | 3 | 16 | 84.21% | cobyla | `RX(q=5) | CNOT(5->4) | CNOT(2->1)` |
| `crlqas__L2_LiH_Equil_6q__L2_LiH_Equil_6q_cobyla_20k_curriculum__seed11111__ep3648__snap0` | 1.053895 | 1.053894 | -0.000001 | 18 | 4 | 14 | 4 | 14 | 77.78% | cobyla | `CNOT(0->3) | RX(q=5) | RX(q=4) | CNOT(4->5)` |
| `crlqas__L2_LiH_Equil_6q__L2_LiH_Equil_6q_cobyla_20k_curriculum__seed11111__ep14100__snap0` | 1.054053 | 1.053894 | -0.000159 | 23 | 2 | 21 | 2 | 21 | 91.30% | cobyla | `RX(q=4) | RY(q=5)` |
| `crlqas__L2_LiH_Equil_6q__L2_LiH_Equil_6q_cobyla_20k_curriculum__seed11111__ep13467__snap0` | 1.053895 | 1.053894 | -0.000001 | 15 | 1 | 14 | 2 | 13 | 86.67% | cobyla | `RY(q=5) | RY(q=4)` |
| `crlqas__L2_LiH_Equil_6q__L2_LiH_Equil_6q_cobyla_20k_curriculum__seed11111__ep1835__snap0` | 1.053911 | 1.053894 | -0.000017 | 15 | 0 | 15 | 2 | 13 | 86.67% | cobyla | `RY(q=4) | RY(q=5)` |
| `crlqas__L2_LiH_Equil_6q__L2_LiH_Equil_6q_cobyla_20k_curriculum__seed11111__ep5201__snap0` | 1.053900 | 1.053894 | -0.000006 | 30 | 5 | 25 | 8 | 22 | 73.33% | cobyla | `RX(q=5) | CNOT(5->2) | CNOT(2->5) | RY(q=5) | CNOT(2->4) | CNOT(5->3) | CNOT(2->3) | RY(q=2)` |
| `crlqas__L2_LiH_Equil_6q__L2_LiH_Equil_6q_cobyla_20k_curriculum__seed11111__ep14464__snap0` | 1.053895 | 1.053894 | -0.000000 | 9 | 1 | 8 | 2 | 7 | 77.78% | cobyla | `RY(q=4) | CNOT(4->5)` |
| `crlqas__L2_LiH_Equil_6q__L2_LiH_Equil_6q_cobyla_20k_curriculum__seed11111__ep5015__snap0` | 1.053895 | 1.053894 | -0.000001 | 10 | 1 | 9 | 2 | 8 | 80.00% | cobyla | `RX(q=4) | RY(q=5)` |
| `crlqas__L2_LiH_Equil_6q__L2_LiH_Equil_6q_cobyla_20k_curriculum__seed11111__ep6660__snap0` | 1.053896 | 1.053894 | -0.000002 | 15 | 2 | 13 | 2 | 13 | 86.67% | cobyla | `RX(q=5) | RX(q=4)` |

**Exact Retained-Structure Matches**

- count=2: `RY(q=4) | CNOT(4->5)`
- count=2: `RX(q=4) | RY(q=5)`
- count=1: `RX(q=5) | CNOT(5->4) | CNOT(2->1)`
- count=1: `CNOT(0->3) | RX(q=5) | RX(q=4) | CNOT(4->5)`
- count=1: `RY(q=5) | RY(q=4)`
- count=1: `RY(q=4) | RY(q=5)`
- count=1: `RX(q=5) | CNOT(5->2) | CNOT(2->5) | RY(q=5) | CNOT(2->4) | CNOT(5->3) | CNOT(2->3) | RY(q=2)`
- count=1: `RX(q=5) | RX(q=4)`

**Common Retained Gate Signatures**

- none
