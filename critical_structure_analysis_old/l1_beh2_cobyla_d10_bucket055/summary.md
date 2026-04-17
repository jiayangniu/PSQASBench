# Summary

- target error threshold: `inherit_from_run_meta`
- selected bucket: `0.55 mHa`
- discovered runs: `1`
- hit episodes in bucket: `9055`
- selected episodes for pruning: `6`

**Anchor Actions**

- `RY(q=5)`: 2484
- `RX(q=5)`: 1751
- `CNOT(4->5)`: 1698

**Per-Episode Pruning Summary**

| Episode | Baseline (mHa) | Retained (mHa) | Δerror (mHa) | Original | Retained | Removed | Redundancy | Optimizer | Retained Gates |
|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| `crlqas__L1_BeH2_STO3G_6q__L1_BeH2_STO3G_6q_cobyla_depth10__seed11111__ep9435` | 0.554445 | 0.554445 | 0.000000 | 2 | 2 | 0 | 0.00% | cobyla | `RX(q=4) | CNOT(4->5)` |
| `crlqas__L1_BeH2_STO3G_6q__L1_BeH2_STO3G_6q_cobyla_depth10__seed11111__ep9093` | 0.554446 | 0.554446 | 0.000000 | 2 | 2 | 0 | 0.00% | cobyla | `RY(q=4) | CNOT(4->5)` |
| `crlqas__L1_BeH2_STO3G_6q__L1_BeH2_STO3G_6q_cobyla_depth10__seed11111__ep8380` | 0.554443 | 0.554443 | 0.000000 | 2 | 2 | 0 | 0.00% | cobyla | `RY(q=4) | RX(q=5)` |
| `crlqas__L1_BeH2_STO3G_6q__L1_BeH2_STO3G_6q_cobyla_depth10__seed11111__ep9471` | 0.554443 | 0.554443 | 0.000000 | 2 | 2 | 0 | 0.00% | cobyla | `RX(q=4) | RX(q=5)` |
| `crlqas__L1_BeH2_STO3G_6q__L1_BeH2_STO3G_6q_cobyla_depth10__seed11111__ep9404` | 0.554447 | 0.554447 | 0.000000 | 2 | 2 | 0 | 0.00% | cobyla | `RX(q=4) | CNOT(4->5)` |
| `crlqas__L1_BeH2_STO3G_6q__L1_BeH2_STO3G_6q_cobyla_depth10__seed11111__ep8040` | 0.554443 | 0.554443 | 0.000000 | 2 | 2 | 0 | 0.00% | cobyla | `RY(q=4) | RY(q=5)` |

**Exact Retained-Structure Matches**

- count=2: `RX(q=4) | CNOT(4->5)`
- count=1: `RY(q=4) | CNOT(4->5)`
- count=1: `RY(q=4) | RX(q=5)`
- count=1: `RX(q=4) | RX(q=5)`
- count=1: `RY(q=4) | RY(q=5)`

**Common Retained Gate Signatures**

- none
