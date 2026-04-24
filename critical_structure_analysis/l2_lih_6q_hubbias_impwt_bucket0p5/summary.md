# Summary

- target error threshold: `saved snapshot events / legacy accept_err fallback`
- selected bucket: `0.5 mHa`
- bucket width: `0.5 mHa`
- discovered runs: `1`
- snapshot events in bucket: `14`
- selected snapshot events for pruning: `10`
- episode sampling: `late_fraction = 1.0`

**Anchor Actions**

- `CNOT(5->4)`: 2
- `CNOT(5->0)`: 2
- `RX(q=4)`: 1

**Per-Snapshot Pruning Summary**

| Snapshot | Baseline (mHa) | Retained (mHa) | Δerror (mHa) | Original | Fixed | Max Steps | Retained | Removed | Redundancy | Optimizer |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `crlqas__L2_LiH_Equil_6q__L2_LiH_Equil_6q_cobyla_20k_curriculum_impwt__seed11111__ep6146__snap0` | 0.231757 | 0.229935 | -0.001822 | 26 | 5 | 21 | 14 | 12 | 46.15% | cobyla |
| `crlqas__L2_LiH_Equil_6q__L2_LiH_Equil_6q_cobyla_20k_curriculum_impwt__seed11111__ep1927__snap0` | 0.268136 | 0.232367 | -0.035769 | 29 | 2 | 27 | 14 | 15 | 51.72% | cobyla |
| `crlqas__L2_LiH_Equil_6q__L2_LiH_Equil_6q_cobyla_20k_curriculum_impwt__seed11111__ep13413__snap0` | 0.378584 | 0.228544 | -0.150040 | 21 | 2 | 19 | 16 | 5 | 23.81% | cobyla |
| `crlqas__L2_LiH_Equil_6q__L2_LiH_Equil_6q_cobyla_20k_curriculum_impwt__seed11111__ep267__snap0` | 0.249475 | 0.228115 | -0.021360 | 38 | 3 | 35 | 21 | 17 | 44.74% | cobyla |
| `crlqas__L2_LiH_Equil_6q__L2_LiH_Equil_6q_cobyla_20k_curriculum_impwt__seed11111__ep13340__snap0` | 0.266429 | 0.228135 | -0.038294 | 13 | 1 | 12 | 7 | 6 | 46.15% | cobyla |
| `crlqas__L2_LiH_Equil_6q__L2_LiH_Equil_6q_cobyla_20k_curriculum_impwt__seed11111__ep7303__snap0` | 0.230980 | 0.228275 | -0.002705 | 19 | 3 | 16 | 9 | 10 | 52.63% | cobyla |
| `crlqas__L2_LiH_Equil_6q__L2_LiH_Equil_6q_cobyla_20k_curriculum_impwt__seed11111__ep3964__snap0` | 0.263525 | 0.228272 | -0.035253 | 15 | 1 | 14 | 7 | 8 | 53.33% | cobyla |
| `crlqas__L2_LiH_Equil_6q__L2_LiH_Equil_6q_cobyla_20k_curriculum_impwt__seed11111__ep9128__snap0` | 0.228789 | 0.232020 | 0.003231 | 40 | 4 | 36 | 27 | 13 | 32.50% | cobyla |
| `crlqas__L2_LiH_Equil_6q__L2_LiH_Equil_6q_cobyla_20k_curriculum_impwt__seed11111__ep1259__snap0` | 0.374413 | 0.228116 | -0.146297 | 17 | 3 | 14 | 10 | 7 | 41.18% | cobyla |
| `crlqas__L2_LiH_Equil_6q__L2_LiH_Equil_6q_cobyla_20k_curriculum_impwt__seed11111__ep4079__snap0` | 0.364065 | 0.228128 | -0.135936 | 30 | 2 | 28 | 7 | 23 | 76.67% | cobyla |

**Exact Retained-Structure Matches**

- count=1: `RX(q=3,θ=+0.070) | RY(q=1,θ=-3.142) | RX(q=4,θ=-0.835) | RY(q=5,θ=+3.142) | RX(q=4,θ=-1.820) | CNOT(3->0) | CNOT(3->2) | CNOT(2->5) | CNOT(2->4) | CNOT(2->3) | RX(q=4,θ=-0.487) | CNOT(0->2) | CNOT(4->1) | RZ(q=5,θ=+1.524)`
- count=1: `RX(q=4,θ=+9.425) | CNOT(4->5) | RX(q=1,θ=+0.070) | RX(q=5,θ=+6.283) | CNOT(1->2) | CNOT(2->5) | RZ(q=2,θ=+7.787) | CNOT(1->4) | RY(q=2,θ=+3.137) | CNOT(5->1) | CNOT(4->2) | RX(q=0,θ=+3.141) | CNOT(5->1) | CNOT(5->0)`
- count=1: `RY(q=3,θ=+3.142) | RY(q=4,θ=+3.212) | CNOT(4->1) | CNOT(4->3) | RY(q=0,θ=-1.591) | CNOT(4->5) | RX(q=5,θ=-3.142) | CNOT(5->4) | RY(q=1,θ=+3.142) | CNOT(5->2) | CNOT(0->2) | CNOT(2->0) | CNOT(5->3) | CNOT(5->4) | RX(q=5,θ=+3.142) | RY(q=2,θ=+1.591)`
- count=1: `RX(q=3,θ=+12.496) | CNOT(3->1) | RY(q=3,θ=+3.142) | CNOT(3->4) | RY(q=2,θ=+3.142) | RY(q=0,θ=+3.142) | CNOT(1->4) | CNOT(3->5) | CNOT(5->4) | CNOT(3->0) | CNOT(5->2) | CNOT(1->5) | CNOT(2->5) | RZ(q=5,θ=+3.849) | RZ(q=4,θ=+2.278) | CNOT(3->4) | RY(q=4,θ=+3.142) | CNOT(3->4) | CNOT(0->3) | CNOT(0->2) | RX(q=3,θ=+3.142)`
- count=1: `RX(q=0,θ=+0.070) | CNOT(0->4) | CNOT(4->1) | RX(q=5,θ=+3.142) | RZ(q=1,θ=-1.575) | RX(q=4,θ=+3.142) | CNOT(0->5)`
- count=1: `RY(q=4,θ=+9.425) | CNOT(4->0) | CNOT(4->1) | RX(q=5,θ=+3.071) | RY(q=4,θ=+3.141) | CNOT(5->1) | RZ(q=1,θ=+1.585) | CNOT(5->4) | CNOT(5->0)`
- count=1: `RX(q=5,θ=+6.354) | CNOT(5->0) | RZ(q=5,θ=+1.585) | RY(q=5,θ=+3.142) | CNOT(5->1) | CNOT(1->4) | RX(q=1,θ=+3.142)`
- count=1: `RX(q=5,θ=+0.070) | CNOT(5->0) | RY(q=4,θ=-3.141) | CNOT(5->4) | CNOT(4->3) | CNOT(5->1) | CNOT(5->2) | CNOT(0->5) | RZ(q=1,θ=-1.640) | CNOT(0->4) | CNOT(3->1) | CNOT(1->5) | CNOT(5->3) | CNOT(4->2) | CNOT(3->4) | CNOT(4->1) | CNOT(1->4) | CNOT(1->2) | CNOT(0->4) | CNOT(1->4) | CNOT(2->4) | CNOT(3->5) | CNOT(3->2) | CNOT(1->3) | CNOT(1->2) | RX(q=2,θ=-3.142) | CNOT(5->4)`
- count=1: `RY(q=2,θ=+15.708) | RX(q=5,θ=+6.213) | CNOT(5->0) | CNOT(0->1) | RY(q=5,θ=+3.142) | RX(q=4,θ=-0.000) | CNOT(1->2) | RZ(q=5,θ=+1.571) | CNOT(5->2) | CNOT(5->4)`
- count=1: `RX(q=1,θ=+3.071) | CNOT(1->5) | RZ(q=1,θ=+1.567) | RY(q=0,θ=+3.142) | CNOT(5->4) | RY(q=1,θ=+3.142) | CNOT(5->0)`

**Common Retained Gate Signatures**

- none
