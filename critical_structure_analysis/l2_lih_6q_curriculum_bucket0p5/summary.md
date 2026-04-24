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

| Snapshot | Baseline (mHa) | Retained (mHa) | Δerror (mHa) | Original | Fixed | Max Steps | Retained | Removed | Redundancy | Optimizer |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `crlqas__L2_LiH_Equil_6q__L2_LiH_Equil_6q_cobyla_20k_curriculum__seed11111__ep568__snap0` | 0.229500 | 0.242415 | 0.012915 | 32 | 0 | 32 | 17 | 15 | 46.88% | cobyla |
| `crlqas__L2_LiH_Equil_6q__L2_LiH_Equil_6q_cobyla_20k_curriculum__seed11111__ep790__snap0` | 0.231046 | 0.228340 | -0.002706 | 34 | 2 | 32 | 17 | 17 | 50.00% | cobyla |
| `crlqas__L2_LiH_Equil_6q__L2_LiH_Equil_6q_cobyla_20k_curriculum__seed11111__ep10240__snap0` | 0.303064 | 0.225707 | -0.077357 | 41 | 2 | 39 | 16 | 25 | 60.98% | cobyla |
| `crlqas__L2_LiH_Equil_6q__L2_LiH_Equil_6q_cobyla_20k_curriculum__seed11111__ep226__snap0` | 0.248218 | 0.297802 | 0.049584 | 36 | 4 | 32 | 24 | 12 | 33.33% | cobyla |
| `crlqas__L2_LiH_Equil_6q__L2_LiH_Equil_6q_cobyla_20k_curriculum__seed11111__ep2171__snap0` | 0.252260 | 0.228115 | -0.024145 | 36 | 4 | 32 | 13 | 23 | 63.89% | cobyla |
| `crlqas__L2_LiH_Equil_6q__L2_LiH_Equil_6q_cobyla_20k_curriculum__seed11111__ep173__snap0` | 0.232883 | 0.350810 | 0.117927 | 31 | 3 | 28 | 17 | 14 | 45.16% | cobyla |
| `crlqas__L2_LiH_Equil_6q__L2_LiH_Equil_6q_cobyla_20k_curriculum__seed11111__ep339__snap0` | 0.260187 | 0.228115 | -0.032072 | 33 | 1 | 32 | 16 | 17 | 51.52% | cobyla |
| `crlqas__L2_LiH_Equil_6q__L2_LiH_Equil_6q_cobyla_20k_curriculum__seed11111__ep9776__snap0` | 0.310967 | 0.242089 | -0.068878 | 22 | 0 | 22 | 18 | 4 | 18.18% | cobyla |
| `crlqas__L2_LiH_Equil_6q__L2_LiH_Equil_6q_cobyla_20k_curriculum__seed11111__ep19396__snap0` | 0.268509 | 0.228118 | -0.040391 | 22 | 0 | 22 | 10 | 12 | 54.55% | cobyla |
| `crlqas__L2_LiH_Equil_6q__L2_LiH_Equil_6q_cobyla_20k_curriculum__seed11111__ep1333__snap0` | 0.260471 | 0.228235 | -0.032236 | 31 | 1 | 30 | 20 | 11 | 35.48% | cobyla |

**Exact Retained-Structure Matches**

- count=1: `RX(q=0,θ=+15.638) | CNOT(0->1) | CNOT(1->3) | CNOT(0->2) | RY(q=3,θ=+9.425) | CNOT(3->0) | RY(q=0,θ=+3.141) | CNOT(1->2) | CNOT(3->4) | CNOT(3->1) | CNOT(3->0) | RX(q=4,θ=+3.142) | CNOT(1->5) | RZ(q=4,θ=+1.703) | CNOT(3->5) | CNOT(4->1) | CNOT(0->3)`
- count=1: `RX(q=0,θ=+0.070) | CNOT(0->5) | CNOT(0->4) | RX(q=3,θ=+9.425) | CNOT(3->0) | CNOT(3->4) | CNOT(3->2) | CNOT(0->5) | CNOT(0->3) | CNOT(4->5) | RZ(q=4,θ=+1.554) | RY(q=0,θ=+3.142) | RY(q=5,θ=+3.142) | RX(q=1,θ=+3.142) | CNOT(4->1) | RX(q=2,θ=+3.141) | CNOT(1->3)`
- count=1: `RY(q=0,θ=+9.416) | CNOT(0->4) | CNOT(0->5) | CNOT(5->2) | CNOT(2->0) | CNOT(5->2) | RX(q=4,θ=+9.425) | RX(q=1,θ=-0.071) | CNOT(1->5) | CNOT(1->0) | RX(q=1,θ=+3.142) | CNOT(1->3) | CNOT(5->3) | CNOT(5->4) | RY(q=1,θ=+3.142) | RZ(q=4,θ=+1.597)`
- count=1: `RX(q=1,θ=+0.068) | CNOT(1->5) | CNOT(5->2) | CNOT(2->3) | RX(q=2,θ=-0.107) | RY(q=5,θ=+3.142) | CNOT(1->0) | CNOT(3->2) | RX(q=2,θ=+3.247) | CNOT(1->5) | RZ(q=1,θ=+1.276) | CNOT(1->2) | CNOT(5->1) | CNOT(3->4) | CNOT(0->5) | RX(q=3,θ=+3.142) | CNOT(0->3) | RX(q=4,θ=-3.142) | CNOT(0->1) | CNOT(5->2) | CNOT(0->3) | CNOT(3->1) | CNOT(4->3) | CNOT(3->2)`
- count=1: `RY(q=1,θ=+3.142) | CNOT(1->0) | RX(q=4,θ=+3.142) | RY(q=4,θ=+0.070) | CNOT(4->2) | CNOT(4->1) | CNOT(4->2) | CNOT(0->4) | CNOT(0->4) | RX(q=0,θ=+3.142) | CNOT(1->5) | RY(q=5,θ=+3.142) | CNOT(1->0)`
- count=1: `RX(q=0,θ=+3.142) | CNOT(0->2) | RX(q=3,θ=+0.064) | RY(q=0,θ=+3.142) | CNOT(3->0) | CNOT(0->5) | RY(q=1,θ=+3.141) | CNOT(1->5) | CNOT(0->4) | CNOT(1->2) | CNOT(3->0) | CNOT(5->3) | CNOT(5->1) | CNOT(1->0) | RZ(q=5,θ=+1.967) | RX(q=3,θ=+3.142) | RX(q=4,θ=+3.141)`
- count=1: `RX(q=0,θ=+3.142) | RX(q=1,θ=+15.708) | CNOT(0->5) | RX(q=3,θ=+3.071) | CNOT(3->1) | RZ(q=3,θ=+4.712) | RY(q=0,θ=+3.142) | CNOT(1->0) | RY(q=4,θ=+3.142) | CNOT(5->3) | CNOT(5->4) | CNOT(3->5) | CNOT(5->4) | CNOT(1->2) | CNOT(1->3) | CNOT(1->2)`
- count=1: `RX(q=2,θ=+3.072) | RY(q=5,θ=+9.425) | CNOT(2->5) | RX(q=1,θ=+6.283) | RX(q=4,θ=+3.141) | CNOT(5->1) | CNOT(1->3) | RY(q=4,θ=+3.141) | CNOT(2->4) | RZ(q=5,θ=+4.583) | RY(q=3,θ=+3.139) | RX(q=3,θ=+3.141) | CNOT(5->0) | RX(q=2,θ=+3.142) | CNOT(0->5) | CNOT(2->3) | CNOT(4->5) | CNOT(1->2)`
- count=1: `RX(q=4,θ=+0.070) | CNOT(4->0) | CNOT(4->5) | CNOT(5->1) | RZ(q=0,θ=-1.573) | RX(q=5,θ=+3.142) | CNOT(4->5) | RX(q=2,θ=-0.000) | CNOT(0->5) | RX(q=4,θ=+3.142)`
- count=1: `RX(q=2,θ=+0.070) | RX(q=3,θ=+0.000) | CNOT(2->3) | CNOT(3->1) | RX(q=5,θ=+3.142) | RY(q=3,θ=-3.142) | RX(q=0,θ=+3.142) | CNOT(0->3) | CNOT(0->1) | CNOT(0->1) | CNOT(3->4) | CNOT(5->2) | CNOT(0->4) | CNOT(0->3) | RX(q=2,θ=+0.000) | CNOT(2->3) | CNOT(1->5) | CNOT(5->0) | RZ(q=4,θ=-1.559) | CNOT(4->2)`

**Common Retained Gate Signatures**

- none
