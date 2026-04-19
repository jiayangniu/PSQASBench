# Summary

- target error threshold: `saved snapshot events / legacy accept_err fallback`
- selected bucket: `0.5 mHa`
- bucket width: `0.5 mHa`
- discovered runs: `1`
- snapshot events in bucket: `22`
- selected snapshot events for pruning: `10`

**Anchor Actions**

- `RY(q=1)`: 2
- `CNOT(3->2)`: 2
- `CNOT(1->0)`: 2

**Per-Snapshot Pruning Summary**

| Snapshot | Baseline (mHa) | Retained (mHa) | Δerror (mHa) | Original | Fixed | Max Steps | Retained | Removed | Redundancy | Optimizer | Retained Gates |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| `crlqas__L2_LiH_Equil_6q__L2_LiH_Equil_6q_cobyla_20k_curriculum__seed11111__ep568__snap0` | 0.229500 | 0.243220 | 0.013720 | 32 | 0 | 32 | 17 | 15 | 46.88% | cobyla | `RX(q=0) | CNOT(0->1) | CNOT(1->3) | CNOT(0->2) | RY(q=3) | CNOT(3->0) | RY(q=0) | CNOT(1->2) | CNOT(3->4) | CNOT(3->1) | CNOT(3->0) | RX(q=4) | CNOT(1->5) | RZ(q=4) | CNOT(3->5) | CNOT(4->1) | CNOT(0->3)` |
| `crlqas__L2_LiH_Equil_6q__L2_LiH_Equil_6q_cobyla_20k_curriculum__seed11111__ep790__snap0` | 0.231046 | 0.228116 | -0.002931 | 34 | 2 | 32 | 17 | 17 | 50.00% | cobyla | `RX(q=0) | CNOT(0->5) | CNOT(0->4) | RX(q=3) | CNOT(3->0) | CNOT(3->4) | CNOT(3->2) | CNOT(0->5) | CNOT(0->3) | CNOT(4->5) | RZ(q=4) | RY(q=0) | RY(q=5) | RX(q=1) | CNOT(4->1) | RX(q=2) | CNOT(1->3)` |
| `crlqas__L2_LiH_Equil_6q__L2_LiH_Equil_6q_cobyla_20k_curriculum__seed11111__ep10240__snap0` | 0.303064 | 0.225148 | -0.077916 | 41 | 2 | 39 | 17 | 24 | 58.54% | cobyla | `RY(q=0) | CNOT(0->4) | CNOT(0->5) | RY(q=4) | CNOT(5->2) | CNOT(2->0) | CNOT(5->2) | RX(q=4) | RX(q=1) | CNOT(1->5) | CNOT(1->0) | RX(q=1) | CNOT(1->3) | CNOT(5->3) | CNOT(5->4) | RY(q=1) | RZ(q=4)` |
| `crlqas__L2_LiH_Equil_6q__L2_LiH_Equil_6q_cobyla_20k_curriculum__seed11111__ep226__snap0` | 0.248218 | 0.234903 | -0.013315 | 36 | 4 | 32 | 24 | 12 | 33.33% | cobyla | `RX(q=1) | CNOT(1->5) | CNOT(5->2) | CNOT(2->3) | RX(q=2) | RY(q=5) | CNOT(1->0) | CNOT(3->2) | RX(q=2) | CNOT(1->5) | RZ(q=1) | CNOT(1->2) | CNOT(5->1) | CNOT(3->4) | CNOT(0->5) | RX(q=3) | CNOT(0->3) | RX(q=4) | CNOT(0->1) | CNOT(5->2) | CNOT(0->3) | CNOT(3->1) | CNOT(4->3) | CNOT(3->2)` |
| `crlqas__L2_LiH_Equil_6q__L2_LiH_Equil_6q_cobyla_20k_curriculum__seed11111__ep2171__snap0` | 0.252260 | 0.228116 | -0.024144 | 36 | 4 | 32 | 13 | 23 | 63.89% | cobyla | `RY(q=1) | CNOT(1->0) | RX(q=4) | RY(q=4) | CNOT(4->2) | CNOT(4->1) | CNOT(4->2) | CNOT(0->4) | CNOT(0->4) | RX(q=0) | CNOT(1->5) | RY(q=5) | CNOT(1->0)` |
| `crlqas__L2_LiH_Equil_6q__L2_LiH_Equil_6q_cobyla_20k_curriculum__seed11111__ep173__snap0` | 0.232883 | 0.368086 | 0.135203 | 31 | 3 | 28 | 17 | 14 | 45.16% | cobyla | `RX(q=0) | CNOT(0->2) | RX(q=3) | RY(q=0) | CNOT(3->0) | CNOT(0->5) | RY(q=1) | CNOT(1->5) | CNOT(0->4) | CNOT(1->2) | CNOT(3->0) | CNOT(5->3) | CNOT(5->1) | CNOT(1->0) | RZ(q=5) | RX(q=3) | RX(q=4)` |
| `crlqas__L2_LiH_Equil_6q__L2_LiH_Equil_6q_cobyla_20k_curriculum__seed11111__ep339__snap0` | 0.260187 | 0.228117 | -0.032070 | 33 | 1 | 32 | 16 | 17 | 51.52% | cobyla | `RX(q=0) | RX(q=1) | CNOT(0->5) | RX(q=3) | CNOT(3->1) | RZ(q=3) | RY(q=0) | CNOT(1->0) | RY(q=4) | CNOT(5->3) | CNOT(5->4) | CNOT(3->5) | CNOT(5->4) | CNOT(1->2) | CNOT(1->3) | CNOT(1->2)` |
| `crlqas__L2_LiH_Equil_6q__L2_LiH_Equil_6q_cobyla_20k_curriculum__seed11111__ep9776__snap0` | 0.310967 | 0.242398 | -0.068569 | 22 | 0 | 22 | 18 | 4 | 18.18% | cobyla | `RX(q=2) | RY(q=5) | CNOT(2->5) | RX(q=1) | RX(q=4) | CNOT(5->1) | CNOT(1->3) | RY(q=4) | CNOT(2->4) | RZ(q=5) | RY(q=3) | RX(q=3) | CNOT(5->0) | RX(q=2) | CNOT(0->5) | CNOT(2->3) | CNOT(4->5) | CNOT(1->2)` |
| `crlqas__L2_LiH_Equil_6q__L2_LiH_Equil_6q_cobyla_20k_curriculum__seed11111__ep19396__snap0` | 0.268509 | 0.238086 | -0.030423 | 22 | 0 | 22 | 10 | 12 | 54.55% | cobyla | `RX(q=4) | CNOT(4->0) | CNOT(4->5) | CNOT(5->1) | RZ(q=0) | RX(q=5) | CNOT(4->5) | RX(q=2) | CNOT(0->5) | RX(q=4)` |
| `crlqas__L2_LiH_Equil_6q__L2_LiH_Equil_6q_cobyla_20k_curriculum__seed11111__ep1333__snap0` | 0.260471 | 0.228119 | -0.032352 | 31 | 1 | 30 | 20 | 11 | 35.48% | cobyla | `RX(q=2) | RX(q=3) | CNOT(2->3) | CNOT(3->1) | RX(q=5) | RY(q=3) | RX(q=0) | CNOT(0->3) | CNOT(0->1) | CNOT(0->1) | CNOT(3->4) | CNOT(5->2) | CNOT(0->4) | CNOT(0->3) | RX(q=2) | CNOT(2->3) | CNOT(1->5) | CNOT(5->0) | RZ(q=4) | CNOT(4->2)` |

**Exact Retained-Structure Matches**

- count=1: `RX(q=0) | CNOT(0->1) | CNOT(1->3) | CNOT(0->2) | RY(q=3) | CNOT(3->0) | RY(q=0) | CNOT(1->2) | CNOT(3->4) | CNOT(3->1) | CNOT(3->0) | RX(q=4) | CNOT(1->5) | RZ(q=4) | CNOT(3->5) | CNOT(4->1) | CNOT(0->3)`
- count=1: `RX(q=0) | CNOT(0->5) | CNOT(0->4) | RX(q=3) | CNOT(3->0) | CNOT(3->4) | CNOT(3->2) | CNOT(0->5) | CNOT(0->3) | CNOT(4->5) | RZ(q=4) | RY(q=0) | RY(q=5) | RX(q=1) | CNOT(4->1) | RX(q=2) | CNOT(1->3)`
- count=1: `RY(q=0) | CNOT(0->4) | CNOT(0->5) | RY(q=4) | CNOT(5->2) | CNOT(2->0) | CNOT(5->2) | RX(q=4) | RX(q=1) | CNOT(1->5) | CNOT(1->0) | RX(q=1) | CNOT(1->3) | CNOT(5->3) | CNOT(5->4) | RY(q=1) | RZ(q=4)`
- count=1: `RX(q=1) | CNOT(1->5) | CNOT(5->2) | CNOT(2->3) | RX(q=2) | RY(q=5) | CNOT(1->0) | CNOT(3->2) | RX(q=2) | CNOT(1->5) | RZ(q=1) | CNOT(1->2) | CNOT(5->1) | CNOT(3->4) | CNOT(0->5) | RX(q=3) | CNOT(0->3) | RX(q=4) | CNOT(0->1) | CNOT(5->2) | CNOT(0->3) | CNOT(3->1) | CNOT(4->3) | CNOT(3->2)`
- count=1: `RY(q=1) | CNOT(1->0) | RX(q=4) | RY(q=4) | CNOT(4->2) | CNOT(4->1) | CNOT(4->2) | CNOT(0->4) | CNOT(0->4) | RX(q=0) | CNOT(1->5) | RY(q=5) | CNOT(1->0)`
- count=1: `RX(q=0) | CNOT(0->2) | RX(q=3) | RY(q=0) | CNOT(3->0) | CNOT(0->5) | RY(q=1) | CNOT(1->5) | CNOT(0->4) | CNOT(1->2) | CNOT(3->0) | CNOT(5->3) | CNOT(5->1) | CNOT(1->0) | RZ(q=5) | RX(q=3) | RX(q=4)`
- count=1: `RX(q=0) | RX(q=1) | CNOT(0->5) | RX(q=3) | CNOT(3->1) | RZ(q=3) | RY(q=0) | CNOT(1->0) | RY(q=4) | CNOT(5->3) | CNOT(5->4) | CNOT(3->5) | CNOT(5->4) | CNOT(1->2) | CNOT(1->3) | CNOT(1->2)`
- count=1: `RX(q=2) | RY(q=5) | CNOT(2->5) | RX(q=1) | RX(q=4) | CNOT(5->1) | CNOT(1->3) | RY(q=4) | CNOT(2->4) | RZ(q=5) | RY(q=3) | RX(q=3) | CNOT(5->0) | RX(q=2) | CNOT(0->5) | CNOT(2->3) | CNOT(4->5) | CNOT(1->2)`
- count=1: `RX(q=4) | CNOT(4->0) | CNOT(4->5) | CNOT(5->1) | RZ(q=0) | RX(q=5) | CNOT(4->5) | RX(q=2) | CNOT(0->5) | RX(q=4)`
- count=1: `RX(q=2) | RX(q=3) | CNOT(2->3) | CNOT(3->1) | RX(q=5) | RY(q=3) | RX(q=0) | CNOT(0->3) | CNOT(0->1) | CNOT(0->1) | CNOT(3->4) | CNOT(5->2) | CNOT(0->4) | CNOT(0->3) | RX(q=2) | CNOT(2->3) | CNOT(1->5) | CNOT(5->0) | RZ(q=4) | CNOT(4->2)`

**Common Retained Gate Signatures**

- none
