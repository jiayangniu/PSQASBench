# Summary

- target error threshold: `saved snapshot events / legacy accept_err fallback`
- selected bucket: `1.05 mHa`
- bucket width: `0.01 mHa`
- discovered runs: `1`
- snapshot events in bucket: `15862`
- selected snapshot events for pruning: `10`

**Anchor Actions**

- `RX(q=5)`: 3022
- `RX(q=4)`: 2718
- `RY(q=4)`: 2195

**Per-Snapshot Pruning Summary**

| Snapshot | Baseline (mHa) | Retained (mHa) | Δerror (mHa) | Original | Fixed | Max Steps | Retained | Removed | Redundancy | Optimizer |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `crlqas__L2_LiH_Equil_6q__L2_LiH_Equil_6q_cobyla_20k_curriculum__seed11111__ep3966__snap0` | 1.053896 | 1.053895 | -0.000001 | 4 | 1 | 3 | 2 | 2 | 50.00% | cobyla |
| `crlqas__L2_LiH_Equil_6q__L2_LiH_Equil_6q_cobyla_20k_curriculum__seed11111__ep2788__snap0` | 1.053898 | 1.053895 | -0.000003 | 8 | 1 | 7 | 2 | 6 | 75.00% | cobyla |
| `crlqas__L2_LiH_Equil_6q__L2_LiH_Equil_6q_cobyla_20k_curriculum__seed11111__ep16546__snap0` | 1.053894 | 1.053894 | -0.000000 | 7 | 2 | 5 | 2 | 5 | 71.43% | cobyla |
| `crlqas__L2_LiH_Equil_6q__L2_LiH_Equil_6q_cobyla_20k_curriculum__seed11111__ep2149__snap0` | 1.053895 | 1.053894 | -0.000001 | 17 | 2 | 15 | 2 | 15 | 88.24% | cobyla |
| `crlqas__L2_LiH_Equil_6q__L2_LiH_Equil_6q_cobyla_20k_curriculum__seed11111__ep17548__snap0` | 1.053894 | 1.053894 | -0.000000 | 5 | 2 | 3 | 2 | 3 | 60.00% | cobyla |
| `crlqas__L2_LiH_Equil_6q__L2_LiH_Equil_6q_cobyla_20k_curriculum__seed11111__ep1575__snap0` | 1.053894 | 1.053894 | -0.000000 | 31 | 3 | 28 | 3 | 28 | 90.32% | cobyla |
| `crlqas__L2_LiH_Equil_6q__L2_LiH_Equil_6q_cobyla_20k_curriculum__seed11111__ep5837__snap0` | 1.053894 | 1.053894 | 0.000000 | 4 | 1 | 3 | 2 | 2 | 50.00% | cobyla |
| `crlqas__L2_LiH_Equil_6q__L2_LiH_Equil_6q_cobyla_20k_curriculum__seed11111__ep11524__snap0` | 1.053894 | 1.053894 | -0.000000 | 3 | 2 | 1 | 2 | 1 | 33.33% | cobyla |
| `crlqas__L2_LiH_Equil_6q__L2_LiH_Equil_6q_cobyla_20k_curriculum__seed11111__ep15421__snap0` | 1.053896 | 1.053897 | 0.000001 | 4 | 1 | 3 | 2 | 2 | 50.00% | cobyla |
| `crlqas__L2_LiH_Equil_6q__L2_LiH_Equil_6q_cobyla_20k_curriculum__seed11111__ep3640__snap0` | 1.053894 | 1.053894 | -0.000001 | 10 | 1 | 9 | 2 | 8 | 80.00% | cobyla |

**Exact Retained-Structure Matches**

- count=2: `RX(q=4,θ=+3.142) | CNOT(4->5)`
- count=1: `RX(q=5,θ=-3.142) | RX(q=4,θ=-3.142)`
- count=1: `RY(q=5,θ=-3.142) | RX(q=4,θ=-3.142)`
- count=1: `RY(q=4,θ=+3.142) | RX(q=5,θ=-3.142)`
- count=1: `RY(q=4,θ=+3.142) | RY(q=5,θ=-3.142) | RX(q=0,θ=-6.283)`
- count=1: `RX(q=4,θ=+3.142) | RY(q=5,θ=+3.142)`
- count=1: `RX(q=5,θ=+3.142) | RY(q=4,θ=+3.142)`
- count=1: `RX(q=5,θ=+3.142) | CNOT(5->4)`
- count=1: `RY(q=4,θ=+3.142) | RY(q=5,θ=-3.142)`

**Common Retained Gate Signatures**

- none
