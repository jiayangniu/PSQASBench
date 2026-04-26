# Summary

- target error threshold: `saved snapshot events / legacy accept_err fallback`
- selected bucket: `0.55 mHa`
- bucket width: `0.01 mHa`
- discovered runs: `1`
- snapshot events in bucket: `17`
- selected snapshot events for pruning: `10`
- episode sampling: `late_fraction = 1.0`

**Anchor Actions**

- `RX(q=1)`: 3
- `CNOT(0->2)`: 2
- `RX(q=2)`: 2

**Per-Snapshot Pruning Summary**

| Snapshot | Baseline (mHa) | Retained (mHa) | Δerror (mHa) | Original | Fixed | Max Steps | Retained | Removed | Redundancy | Optimizer |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `tfqas__L1_BeH2_STO3G_6q__L1_BeH2_STO3G_6q_depth10__seed11111__ep27__snap0` | 0.554446 | 0.554444 | -0.000002 | 10 | 1 | 9 | 2 | 8 | 80.00% | cobyla |
| `tfqas__L1_BeH2_STO3G_6q__L1_BeH2_STO3G_6q_depth10__seed11111__ep12__snap0` | 0.554445 | 0.554444 | -0.000000 | 10 | 3 | 7 | 3 | 7 | 70.00% | cobyla |
| `tfqas__L1_BeH2_STO3G_6q__L1_BeH2_STO3G_6q_depth10__seed11111__ep46__snap0` | 0.554444 | 0.554443 | -0.000000 | 10 | 1 | 9 | 2 | 8 | 80.00% | cobyla |
| `tfqas__L1_BeH2_STO3G_6q__L1_BeH2_STO3G_6q_depth10__seed11111__ep39__snap0` | 0.554444 | 0.554444 | -0.000000 | 10 | 0 | 10 | 2 | 8 | 80.00% | cobyla |
| `tfqas__L1_BeH2_STO3G_6q__L1_BeH2_STO3G_6q_depth10__seed11111__ep98__snap0` | 0.554447 | 0.554444 | -0.000004 | 10 | 2 | 8 | 2 | 8 | 80.00% | cobyla |
| `tfqas__L1_BeH2_STO3G_6q__L1_BeH2_STO3G_6q_depth10__seed11111__ep83__snap0` | 0.554443 | 0.554443 | -0.000000 | 10 | 0 | 10 | 2 | 8 | 80.00% | cobyla |
| `tfqas__L1_BeH2_STO3G_6q__L1_BeH2_STO3G_6q_depth10__seed11111__ep51__snap0` | 0.554445 | 0.554443 | -0.000001 | 10 | 0 | 10 | 2 | 8 | 80.00% | cobyla |
| `tfqas__L1_BeH2_STO3G_6q__L1_BeH2_STO3G_6q_depth10__seed11111__ep2__snap0` | 0.554444 | 0.554443 | -0.000000 | 10 | 0 | 10 | 2 | 8 | 80.00% | cobyla |
| `tfqas__L1_BeH2_STO3G_6q__L1_BeH2_STO3G_6q_depth10__seed11111__ep72__snap0` | 0.554444 | 0.554443 | -0.000001 | 10 | 0 | 10 | 2 | 8 | 80.00% | cobyla |
| `tfqas__L1_BeH2_STO3G_6q__L1_BeH2_STO3G_6q_depth10__seed11111__ep62__snap0` | 0.554443 | 0.554443 | 0.000000 | 10 | 1 | 9 | 2 | 8 | 80.00% | cobyla |

**Exact Retained-Structure Matches**

- count=1: `RX(q=4,θ=+3.142) | RY(q=5,θ=+3.142)`
- count=1: `RX(q=4,θ=-3.142) | RY(q=5,θ=+3.142)`
- count=1: `RX(q=4,θ=+3.142) | RY(q=5,θ=-3.142)`
- count=1: `RX(q=4,θ=+3.142) | CNOT(4->5)`
- count=1: `RX(q=4,θ=-3.142) | CNOT(4->5)`
- count=1: `RX(q=5,θ=-3.142) | RY(q=4,θ=+3.142)`
- count=1: `RX(q=5,θ=+3.142) | RY(q=4,θ=+3.142)`
- count=1: `RX(q=1,θ=-6.283) | RX(q=4,θ=-3.142) | CNOT(4->5)`
- count=1: `RY(q=5,θ=+3.142) | RY(q=4,θ=+3.142)`
- count=1: `RX(q=4,θ=-3.142) | RX(q=5,θ=+3.142)`

**Common Retained Gate Signatures**

- none
