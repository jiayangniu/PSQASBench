# Summary

- target error threshold: `saved snapshot events / legacy accept_err fallback`
- selected bucket: `0.55 mHa`
- bucket width: `0.01 mHa`
- discovered runs: `1`
- snapshot events in bucket: `9434`
- selected snapshot events for pruning: `10`

**Anchor Actions**

- `RX(q=5)`: 1956
- `RY(q=5)`: 1767
- `CNOT(5->4)`: 1551

**Per-Snapshot Pruning Summary**

| Snapshot | Baseline (mHa) | Retained (mHa) | Δerror (mHa) | Original | Fixed | Max Steps | Retained | Removed | Redundancy | Optimizer |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `crlqas__L1_BeH2_STO3G_6q__L1_BeH2_STO3G_6q_cobyla_20k_depth50__seed11111__ep6695__snap0` | 0.554444 | 0.554444 | 0.000000 | 2 | 1 | 1 | 2 | 0 | 0.00% | cobyla |
| `crlqas__L1_BeH2_STO3G_6q__L1_BeH2_STO3G_6q_cobyla_20k_depth50__seed11111__ep3770__snap0` | 0.554443 | 0.554444 | 0.000000 | 5 | 1 | 4 | 2 | 3 | 60.00% | cobyla |
| `crlqas__L1_BeH2_STO3G_6q__L1_BeH2_STO3G_6q_cobyla_20k_depth50__seed11111__ep4080__snap0` | 0.554446 | 0.554446 | 0.000000 | 2 | 2 | 0 | 2 | 0 | 0.00% | cobyla |
| `crlqas__L1_BeH2_STO3G_6q__L1_BeH2_STO3G_6q_cobyla_20k_depth50__seed11111__ep593__snap0` | 0.554444 | 0.554443 | -0.000001 | 5 | 1 | 4 | 2 | 3 | 60.00% | cobyla |
| `crlqas__L1_BeH2_STO3G_6q__L1_BeH2_STO3G_6q_cobyla_20k_depth50__seed11111__ep5833__snap0` | 0.554444 | 0.554444 | 0.000000 | 2 | 1 | 1 | 2 | 0 | 0.00% | cobyla |
| `crlqas__L1_BeH2_STO3G_6q__L1_BeH2_STO3G_6q_cobyla_20k_depth50__seed11111__ep8498__snap0` | 0.554443 | 0.554443 | 0.000000 | 2 | 1 | 1 | 2 | 0 | 0.00% | cobyla |
| `crlqas__L1_BeH2_STO3G_6q__L1_BeH2_STO3G_6q_cobyla_20k_depth50__seed11111__ep3501__snap0` | 0.554444 | 0.554444 | 0.000000 | 3 | 1 | 2 | 2 | 1 | 33.33% | cobyla |
| `crlqas__L1_BeH2_STO3G_6q__L1_BeH2_STO3G_6q_cobyla_20k_depth50__seed11111__ep2632__snap0` | 0.554444 | 0.554443 | -0.000000 | 5 | 2 | 3 | 2 | 3 | 60.00% | cobyla |
| `crlqas__L1_BeH2_STO3G_6q__L1_BeH2_STO3G_6q_cobyla_20k_depth50__seed11111__ep1722__snap0` | 0.554444 | 0.554444 | 0.000000 | 2 | 1 | 1 | 2 | 0 | 0.00% | cobyla |
| `crlqas__L1_BeH2_STO3G_6q__L1_BeH2_STO3G_6q_cobyla_20k_depth50__seed11111__ep5262__snap0` | 0.554444 | 0.554444 | 0.000000 | 2 | 1 | 1 | 2 | 0 | 0.00% | cobyla |

**Exact Retained-Structure Matches**

- count=2: `RY(q=4,θ=+3.142) | RX(q=5,θ=+3.142)`
- count=1: `RX(q=5,θ=-3.142) | RY(q=4,θ=+3.142)`
- count=1: `RX(q=5,θ=+3.142) | RY(q=4,θ=+3.142)`
- count=1: `RX(q=4,θ=-3.142) | RX(q=5,θ=-3.142)`
- count=1: `RX(q=4,θ=+3.142) | RX(q=5,θ=+3.141)`
- count=1: `RX(q=5,θ=+3.142) | CNOT(5->4)`
- count=1: `RX(q=4,θ=+3.142) | RY(q=5,θ=+3.142)`
- count=1: `RX(q=5,θ=+3.142) | RX(q=4,θ=+3.142)`
- count=1: `RY(q=4,θ=+3.142) | RY(q=5,θ=+3.142)`

**Common Retained Gate Signatures**

- none
