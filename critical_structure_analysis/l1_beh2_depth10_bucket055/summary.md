# Summary

- target error threshold: `saved snapshot events / legacy accept_err fallback`
- selected bucket: `0.55 mHa`
- bucket width: `0.01 mHa`
- discovered runs: `1`
- snapshot events in bucket: `8997`
- selected snapshot events for pruning: `10`

**Anchor Actions**

- `RX(q=5)`: 2449
- `RY(q=5)`: 1652
- `CNOT(4->5)`: 1551

**Per-Snapshot Pruning Summary**

| Snapshot | Baseline (mHa) | Retained (mHa) | Δerror (mHa) | Original | Fixed | Max Steps | Retained | Removed | Redundancy | Optimizer |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `crlqas__L1_BeH2_STO3G_6q__L1_BeH2_STO3G_6q_cobyla_20k_depth10__seed11111__ep7215__snap0` | 0.554443 | 0.554443 | 0.000000 | 2 | 1 | 1 | 2 | 0 | 0.00% | cobyla |
| `crlqas__L1_BeH2_STO3G_6q__L1_BeH2_STO3G_6q_cobyla_20k_depth10__seed11111__ep9320__snap0` | 0.554443 | 0.554443 | -0.000000 | 3 | 1 | 2 | 2 | 1 | 33.33% | cobyla |
| `crlqas__L1_BeH2_STO3G_6q__L1_BeH2_STO3G_6q_cobyla_20k_depth10__seed11111__ep6417__snap0` | 0.554446 | 0.554446 | 0.000000 | 2 | 1 | 1 | 2 | 0 | 0.00% | cobyla |
| `crlqas__L1_BeH2_STO3G_6q__L1_BeH2_STO3G_6q_cobyla_20k_depth10__seed11111__ep8032__snap0` | 0.554444 | 0.554444 | 0.000000 | 2 | 1 | 1 | 2 | 0 | 0.00% | cobyla |
| `crlqas__L1_BeH2_STO3G_6q__L1_BeH2_STO3G_6q_cobyla_20k_depth10__seed11111__ep3985__snap0` | 0.554444 | 0.554443 | -0.000000 | 4 | 1 | 3 | 2 | 2 | 50.00% | cobyla |
| `crlqas__L1_BeH2_STO3G_6q__L1_BeH2_STO3G_6q_cobyla_20k_depth10__seed11111__ep1147__snap0` | 0.554443 | 0.554444 | 0.000000 | 3 | 1 | 2 | 2 | 1 | 33.33% | cobyla |
| `crlqas__L1_BeH2_STO3G_6q__L1_BeH2_STO3G_6q_cobyla_20k_depth10__seed11111__ep6514__snap0` | 0.554444 | 0.554444 | -0.000000 | 3 | 1 | 2 | 2 | 1 | 33.33% | cobyla |
| `crlqas__L1_BeH2_STO3G_6q__L1_BeH2_STO3G_6q_cobyla_20k_depth10__seed11111__ep3697__snap0` | 0.554446 | 0.554446 | 0.000000 | 2 | 1 | 1 | 2 | 0 | 0.00% | cobyla |
| `crlqas__L1_BeH2_STO3G_6q__L1_BeH2_STO3G_6q_cobyla_20k_depth10__seed11111__ep6335__snap0` | 0.554444 | 0.554443 | -0.000000 | 3 | 1 | 2 | 2 | 1 | 33.33% | cobyla |
| `crlqas__L1_BeH2_STO3G_6q__L1_BeH2_STO3G_6q_cobyla_20k_depth10__seed11111__ep1473__snap0` | 0.554443 | 0.554443 | -0.000000 | 6 | 1 | 5 | 2 | 4 | 66.67% | cobyla |

**Exact Retained-Structure Matches**

- count=1: `RX(q=4,θ=-3.142) | RX(q=5,θ=-3.142)`
- count=1: `RX(q=4,θ=+3.142) | RX(q=5,θ=+3.142)`
- count=1: `RX(q=4,θ=+3.142) | RX(q=5,θ=-3.142)`
- count=1: `RX(q=5,θ=+3.142) | CNOT(5->4)`
- count=1: `RX(q=5,θ=-3.142) | CNOT(5->4)`
- count=1: `RY(q=5,θ=+3.142) | RX(q=4,θ=+3.142)`
- count=1: `RY(q=4,θ=+3.142) | RX(q=5,θ=-3.142)`
- count=1: `RY(q=4,θ=+3.142) | RY(q=5,θ=-3.142)`
- count=1: `RX(q=4,θ=+3.142) | RY(q=5,θ=+3.142)`
- count=1: `RX(q=4,θ=+3.142) | CNOT(4->5)`

**Common Retained Gate Signatures**

- none
