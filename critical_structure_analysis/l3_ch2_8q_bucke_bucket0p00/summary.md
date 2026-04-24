# Summary

- target error threshold: `saved snapshot events / legacy accept_err fallback`
- selected bucket: `0.00 mHa`
- bucket width: `0.01 mHa`
- discovered runs: `1`
- snapshot events in bucket: `11229`
- selected snapshot events for pruning: `100`

**Anchor Actions**

- `CNOT(1->5)`: 9157
- `CNOT(0->4)`: 1328
- `RX(q=7)`: 200

**Per-Snapshot Pruning Summary**

| Snapshot | Baseline (mHa) | Retained (mHa) | Δerror (mHa) | Original | Fixed | Max Steps | Retained | Removed | Redundancy | Optimizer |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `crlqas__L3_CH2_Singlet_8q__L3_CH2_Singlet_8q_rotosolve_s2_20k__seed11111__ep14467__snap0` | 0.000000 | 0.000000 | 0.000000 | 10 | 2 | 8 | 4 | 6 | 60.00% | rotosolve |
| `crlqas__L3_CH2_Singlet_8q__L3_CH2_Singlet_8q_rotosolve_s2_20k__seed11111__ep8081__snap0` | 0.000000 | 0.000000 | 0.000000 | 7 | 2 | 5 | 4 | 3 | 42.86% | rotosolve |
| `crlqas__L3_CH2_Singlet_8q__L3_CH2_Singlet_8q_rotosolve_s2_20k__seed11111__ep8752__snap0` | 0.000000 | 0.000000 | 0.000000 | 7 | 0 | 7 | 4 | 3 | 42.86% | rotosolve |
| `crlqas__L3_CH2_Singlet_8q__L3_CH2_Singlet_8q_rotosolve_s2_20k__seed11111__ep11022__snap0` | 0.000000 | 0.000000 | 0.000000 | 5 | 1 | 4 | 4 | 1 | 20.00% | rotosolve |
| `crlqas__L3_CH2_Singlet_8q__L3_CH2_Singlet_8q_rotosolve_s2_20k__seed11111__ep16998__snap0` | 0.000000 | 0.000000 | 0.000000 | 5 | 2 | 3 | 4 | 1 | 20.00% | rotosolve |
| `crlqas__L3_CH2_Singlet_8q__L3_CH2_Singlet_8q_rotosolve_s2_20k__seed11111__ep18029__snap0` | 0.000000 | 0.000000 | 0.000000 | 6 | 2 | 4 | 4 | 2 | 33.33% | rotosolve |
| `crlqas__L3_CH2_Singlet_8q__L3_CH2_Singlet_8q_rotosolve_s2_20k__seed11111__ep12828__snap0` | 0.000000 | 0.000000 | 0.000000 | 4 | 1 | 3 | 4 | 0 | 0.00% | rotosolve |
| `crlqas__L3_CH2_Singlet_8q__L3_CH2_Singlet_8q_rotosolve_s2_20k__seed11111__ep18776__snap0` | 0.000000 | 0.000000 | 0.000000 | 5 | 2 | 3 | 4 | 1 | 20.00% | rotosolve |
| `crlqas__L3_CH2_Singlet_8q__L3_CH2_Singlet_8q_rotosolve_s2_20k__seed11111__ep13899__snap0` | 0.000000 | 0.000000 | 0.000000 | 7 | 2 | 5 | 4 | 3 | 42.86% | rotosolve |
| `crlqas__L3_CH2_Singlet_8q__L3_CH2_Singlet_8q_rotosolve_s2_20k__seed11111__ep15828__snap0` | 0.000000 | 0.000000 | 0.000000 | 5 | 1 | 4 | 4 | 1 | 20.00% | rotosolve |
| `crlqas__L3_CH2_Singlet_8q__L3_CH2_Singlet_8q_rotosolve_s2_20k__seed11111__ep9195__snap0` | 0.000000 | 0.000000 | 0.000000 | 13 | 3 | 10 | 4 | 9 | 69.23% | rotosolve |
| `crlqas__L3_CH2_Singlet_8q__L3_CH2_Singlet_8q_rotosolve_s2_20k__seed11111__ep19633__snap0` | 0.000000 | 0.000000 | 0.000000 | 6 | 1 | 5 | 4 | 2 | 33.33% | rotosolve |
| `crlqas__L3_CH2_Singlet_8q__L3_CH2_Singlet_8q_rotosolve_s2_20k__seed11111__ep13699__snap0` | 0.000000 | 0.000000 | 0.000000 | 6 | 2 | 4 | 4 | 2 | 33.33% | rotosolve |
| `crlqas__L3_CH2_Singlet_8q__L3_CH2_Singlet_8q_rotosolve_s2_20k__seed11111__ep11302__snap0` | 0.000000 | 0.000000 | 0.000000 | 7 | 2 | 5 | 4 | 3 | 42.86% | rotosolve |
| `crlqas__L3_CH2_Singlet_8q__L3_CH2_Singlet_8q_rotosolve_s2_20k__seed11111__ep16549__snap0` | 0.000000 | 0.000000 | 0.000000 | 8 | 1 | 7 | 4 | 4 | 50.00% | rotosolve |
| `crlqas__L3_CH2_Singlet_8q__L3_CH2_Singlet_8q_rotosolve_s2_20k__seed11111__ep12759__snap0` | 0.000000 | 0.000000 | 0.000000 | 4 | 1 | 3 | 4 | 0 | 0.00% | rotosolve |
| `crlqas__L3_CH2_Singlet_8q__L3_CH2_Singlet_8q_rotosolve_s2_20k__seed11111__ep10182__snap0` | 0.000000 | 0.000000 | 0.000000 | 7 | 2 | 5 | 4 | 3 | 42.86% | rotosolve |
| `crlqas__L3_CH2_Singlet_8q__L3_CH2_Singlet_8q_rotosolve_s2_20k__seed11111__ep13909__snap0` | 0.000000 | 0.000000 | 0.000000 | 5 | 2 | 3 | 4 | 1 | 20.00% | rotosolve |
| `crlqas__L3_CH2_Singlet_8q__L3_CH2_Singlet_8q_rotosolve_s2_20k__seed11111__ep10882__snap0` | 0.000000 | 0.000000 | 0.000000 | 9 | 2 | 7 | 4 | 5 | 55.56% | rotosolve |
| `crlqas__L3_CH2_Singlet_8q__L3_CH2_Singlet_8q_rotosolve_s2_20k__seed11111__ep19802__snap0` | 0.000000 | 0.000000 | 0.000000 | 6 | 2 | 4 | 4 | 2 | 33.33% | rotosolve |
| `crlqas__L3_CH2_Singlet_8q__L3_CH2_Singlet_8q_rotosolve_s2_20k__seed11111__ep11589__snap0` | 0.000000 | 0.000000 | 0.000000 | 6 | 2 | 4 | 4 | 2 | 33.33% | rotosolve |
| `crlqas__L3_CH2_Singlet_8q__L3_CH2_Singlet_8q_rotosolve_s2_20k__seed11111__ep18265__snap0` | 0.000000 | 0.000000 | 0.000000 | 6 | 1 | 5 | 4 | 2 | 33.33% | rotosolve |
| `crlqas__L3_CH2_Singlet_8q__L3_CH2_Singlet_8q_rotosolve_s2_20k__seed11111__ep9823__snap0` | 0.000000 | 0.000000 | 0.000000 | 4 | 2 | 2 | 4 | 0 | 0.00% | rotosolve |
| `crlqas__L3_CH2_Singlet_8q__L3_CH2_Singlet_8q_rotosolve_s2_20k__seed11111__ep18124__snap0` | 0.000000 | 0.000000 | 0.000000 | 5 | 1 | 4 | 4 | 1 | 20.00% | rotosolve |
| `crlqas__L3_CH2_Singlet_8q__L3_CH2_Singlet_8q_rotosolve_s2_20k__seed11111__ep15363__snap0` | 0.000000 | 0.000000 | 0.000000 | 10 | 1 | 9 | 4 | 6 | 60.00% | rotosolve |
| `crlqas__L3_CH2_Singlet_8q__L3_CH2_Singlet_8q_rotosolve_s2_20k__seed11111__ep17524__snap0` | 0.000000 | 0.000000 | 0.000000 | 6 | 1 | 5 | 4 | 2 | 33.33% | rotosolve |
| `crlqas__L3_CH2_Singlet_8q__L3_CH2_Singlet_8q_rotosolve_s2_20k__seed11111__ep12216__snap0` | 0.000000 | 0.000000 | 0.000000 | 6 | 2 | 4 | 4 | 2 | 33.33% | rotosolve |
| `crlqas__L3_CH2_Singlet_8q__L3_CH2_Singlet_8q_rotosolve_s2_20k__seed11111__ep17357__snap0` | 0.000000 | 0.000000 | 0.000000 | 9 | 0 | 9 | 4 | 5 | 55.56% | rotosolve |
| `crlqas__L3_CH2_Singlet_8q__L3_CH2_Singlet_8q_rotosolve_s2_20k__seed11111__ep7374__snap0` | 0.000000 | 0.000000 | 0.000000 | 9 | 2 | 7 | 4 | 5 | 55.56% | rotosolve |
| `crlqas__L3_CH2_Singlet_8q__L3_CH2_Singlet_8q_rotosolve_s2_20k__seed11111__ep7751__snap0` | 0.000000 | 0.000000 | 0.000000 | 6 | 2 | 4 | 4 | 2 | 33.33% | rotosolve |
| `crlqas__L3_CH2_Singlet_8q__L3_CH2_Singlet_8q_rotosolve_s2_20k__seed11111__ep19516__snap0` | 0.000000 | 0.000000 | 0.000000 | 5 | 2 | 3 | 4 | 1 | 20.00% | rotosolve |
| `crlqas__L3_CH2_Singlet_8q__L3_CH2_Singlet_8q_rotosolve_s2_20k__seed11111__ep8596__snap0` | 0.000000 | 0.000000 | 0.000000 | 6 | 2 | 4 | 4 | 2 | 33.33% | rotosolve |
| `crlqas__L3_CH2_Singlet_8q__L3_CH2_Singlet_8q_rotosolve_s2_20k__seed11111__ep16735__snap0` | 0.000000 | 0.000000 | 0.000000 | 7 | 1 | 6 | 4 | 3 | 42.86% | rotosolve |
| `crlqas__L3_CH2_Singlet_8q__L3_CH2_Singlet_8q_rotosolve_s2_20k__seed11111__ep18010__snap0` | 0.000000 | 0.000000 | 0.000000 | 6 | 2 | 4 | 4 | 2 | 33.33% | rotosolve |
| `crlqas__L3_CH2_Singlet_8q__L3_CH2_Singlet_8q_rotosolve_s2_20k__seed11111__ep14570__snap0` | 0.000000 | 0.000000 | 0.000000 | 5 | 1 | 4 | 4 | 1 | 20.00% | rotosolve |
| `crlqas__L3_CH2_Singlet_8q__L3_CH2_Singlet_8q_rotosolve_s2_20k__seed11111__ep10157__snap0` | 0.000000 | 0.000000 | 0.000000 | 5 | 1 | 4 | 4 | 1 | 20.00% | rotosolve |
| `crlqas__L3_CH2_Singlet_8q__L3_CH2_Singlet_8q_rotosolve_s2_20k__seed11111__ep7028__snap0` | 0.000000 | 0.000000 | 0.000000 | 7 | 1 | 6 | 4 | 3 | 42.86% | rotosolve |
| `crlqas__L3_CH2_Singlet_8q__L3_CH2_Singlet_8q_rotosolve_s2_20k__seed11111__ep9339__snap0` | 0.000000 | 0.000000 | 0.000000 | 6 | 2 | 4 | 4 | 2 | 33.33% | rotosolve |
| `crlqas__L3_CH2_Singlet_8q__L3_CH2_Singlet_8q_rotosolve_s2_20k__seed11111__ep16051__snap0` | 0.000000 | 0.000000 | 0.000000 | 6 | 1 | 5 | 4 | 2 | 33.33% | rotosolve |
| `crlqas__L3_CH2_Singlet_8q__L3_CH2_Singlet_8q_rotosolve_s2_20k__seed11111__ep15100__snap0` | 0.000000 | 0.000000 | 0.000000 | 5 | 1 | 4 | 4 | 1 | 20.00% | rotosolve |
| `crlqas__L3_CH2_Singlet_8q__L3_CH2_Singlet_8q_rotosolve_s2_20k__seed11111__ep19498__snap0` | 0.000000 | 0.000000 | 0.000000 | 5 | 2 | 3 | 4 | 1 | 20.00% | rotosolve |
| `crlqas__L3_CH2_Singlet_8q__L3_CH2_Singlet_8q_rotosolve_s2_20k__seed11111__ep15635__snap0` | 0.000000 | 0.000000 | 0.000000 | 5 | 1 | 4 | 4 | 1 | 20.00% | rotosolve |
| `crlqas__L3_CH2_Singlet_8q__L3_CH2_Singlet_8q_rotosolve_s2_20k__seed11111__ep9375__snap0` | 0.000000 | 0.000000 | 0.000000 | 7 | 2 | 5 | 4 | 3 | 42.86% | rotosolve |
| `crlqas__L3_CH2_Singlet_8q__L3_CH2_Singlet_8q_rotosolve_s2_20k__seed11111__ep14687__snap0` | 0.000000 | 0.000000 | 0.000000 | 5 | 1 | 4 | 4 | 1 | 20.00% | rotosolve |
| `crlqas__L3_CH2_Singlet_8q__L3_CH2_Singlet_8q_rotosolve_s2_20k__seed11111__ep11662__snap0` | 0.000000 | 0.000000 | 0.000000 | 7 | 1 | 6 | 4 | 3 | 42.86% | rotosolve |
| `crlqas__L3_CH2_Singlet_8q__L3_CH2_Singlet_8q_rotosolve_s2_20k__seed11111__ep238__snap0` | 0.000000 | 0.000000 | 0.000000 | 43 | 3 | 40 | 8 | 35 | 81.40% | rotosolve |
| `crlqas__L3_CH2_Singlet_8q__L3_CH2_Singlet_8q_rotosolve_s2_20k__seed11111__ep7182__snap0` | 0.000000 | 0.000000 | 0.000000 | 9 | 1 | 8 | 4 | 5 | 55.56% | rotosolve |
| `crlqas__L3_CH2_Singlet_8q__L3_CH2_Singlet_8q_rotosolve_s2_20k__seed11111__ep18556__snap0` | 0.000000 | 0.000000 | 0.000000 | 6 | 1 | 5 | 4 | 2 | 33.33% | rotosolve |
| `crlqas__L3_CH2_Singlet_8q__L3_CH2_Singlet_8q_rotosolve_s2_20k__seed11111__ep6318__snap0` | 0.000000 | 0.000000 | 0.000000 | 6 | 1 | 5 | 4 | 2 | 33.33% | rotosolve |
| `crlqas__L3_CH2_Singlet_8q__L3_CH2_Singlet_8q_rotosolve_s2_20k__seed11111__ep10825__snap0` | 0.000000 | 0.000000 | 0.000000 | 5 | 1 | 4 | 4 | 1 | 20.00% | rotosolve |
| `crlqas__L3_CH2_Singlet_8q__L3_CH2_Singlet_8q_rotosolve_s2_20k__seed11111__ep11157__snap0` | 0.000000 | 0.000000 | 0.000000 | 9 | 2 | 7 | 4 | 5 | 55.56% | rotosolve |
| `crlqas__L3_CH2_Singlet_8q__L3_CH2_Singlet_8q_rotosolve_s2_20k__seed11111__ep19832__snap0` | 0.000000 | 0.000000 | 0.000000 | 8 | 2 | 6 | 4 | 4 | 50.00% | rotosolve |
| `crlqas__L3_CH2_Singlet_8q__L3_CH2_Singlet_8q_rotosolve_s2_20k__seed11111__ep17468__snap0` | 0.000000 | 0.000000 | 0.000000 | 7 | 1 | 6 | 4 | 3 | 42.86% | rotosolve |
| `crlqas__L3_CH2_Singlet_8q__L3_CH2_Singlet_8q_rotosolve_s2_20k__seed11111__ep8689__snap0` | 0.000000 | 0.000000 | 0.000000 | 7 | 2 | 5 | 4 | 3 | 42.86% | rotosolve |
| `crlqas__L3_CH2_Singlet_8q__L3_CH2_Singlet_8q_rotosolve_s2_20k__seed11111__ep12140__snap0` | 0.000000 | 0.000000 | 0.000000 | 5 | 1 | 4 | 4 | 1 | 20.00% | rotosolve |
| `crlqas__L3_CH2_Singlet_8q__L3_CH2_Singlet_8q_rotosolve_s2_20k__seed11111__ep14438__snap0` | 0.000000 | 0.000000 | 0.000000 | 6 | 2 | 4 | 4 | 2 | 33.33% | rotosolve |
| `crlqas__L3_CH2_Singlet_8q__L3_CH2_Singlet_8q_rotosolve_s2_20k__seed11111__ep17062__snap0` | 0.000000 | 0.000000 | 0.000000 | 6 | 1 | 5 | 4 | 2 | 33.33% | rotosolve |
| `crlqas__L3_CH2_Singlet_8q__L3_CH2_Singlet_8q_rotosolve_s2_20k__seed11111__ep18630__snap0` | 0.000000 | 0.000000 | 0.000000 | 6 | 1 | 5 | 4 | 2 | 33.33% | rotosolve |
| `crlqas__L3_CH2_Singlet_8q__L3_CH2_Singlet_8q_rotosolve_s2_20k__seed11111__ep17891__snap0` | 0.000000 | 0.000000 | 0.000000 | 6 | 1 | 5 | 4 | 2 | 33.33% | rotosolve |
| `crlqas__L3_CH2_Singlet_8q__L3_CH2_Singlet_8q_rotosolve_s2_20k__seed11111__ep10177__snap0` | 0.000000 | 0.000000 | 0.000000 | 5 | 2 | 3 | 4 | 1 | 20.00% | rotosolve |
| `crlqas__L3_CH2_Singlet_8q__L3_CH2_Singlet_8q_rotosolve_s2_20k__seed11111__ep17195__snap0` | 0.000000 | 0.000000 | 0.000000 | 6 | 1 | 5 | 4 | 2 | 33.33% | rotosolve |
| `crlqas__L3_CH2_Singlet_8q__L3_CH2_Singlet_8q_rotosolve_s2_20k__seed11111__ep8316__snap0` | 0.000000 | 0.000000 | 0.000000 | 5 | 2 | 3 | 4 | 1 | 20.00% | rotosolve |
| `crlqas__L3_CH2_Singlet_8q__L3_CH2_Singlet_8q_rotosolve_s2_20k__seed11111__ep18355__snap0` | 0.000000 | 0.000000 | 0.000000 | 7 | 1 | 6 | 4 | 3 | 42.86% | rotosolve |
| `crlqas__L3_CH2_Singlet_8q__L3_CH2_Singlet_8q_rotosolve_s2_20k__seed11111__ep9260__snap0` | 0.000000 | 0.000000 | 0.000000 | 7 | 2 | 5 | 4 | 3 | 42.86% | rotosolve |
| `crlqas__L3_CH2_Singlet_8q__L3_CH2_Singlet_8q_rotosolve_s2_20k__seed11111__ep16597__snap0` | 0.000000 | 0.000000 | 0.000000 | 6 | 2 | 4 | 4 | 2 | 33.33% | rotosolve |
| `crlqas__L3_CH2_Singlet_8q__L3_CH2_Singlet_8q_rotosolve_s2_20k__seed11111__ep10414__snap0` | 0.000000 | 0.000000 | 0.000000 | 8 | 2 | 6 | 4 | 4 | 50.00% | rotosolve |
| `crlqas__L3_CH2_Singlet_8q__L3_CH2_Singlet_8q_rotosolve_s2_20k__seed11111__ep9997__snap0` | 0.000000 | 0.000000 | 0.000000 | 7 | 2 | 5 | 4 | 3 | 42.86% | rotosolve |
| `crlqas__L3_CH2_Singlet_8q__L3_CH2_Singlet_8q_rotosolve_s2_20k__seed11111__ep12218__snap0` | 0.000000 | 0.000000 | 0.000000 | 4 | 1 | 3 | 4 | 0 | 0.00% | rotosolve |
| `crlqas__L3_CH2_Singlet_8q__L3_CH2_Singlet_8q_rotosolve_s2_20k__seed11111__ep11024__snap0` | 0.000000 | 0.000000 | 0.000000 | 5 | 0 | 5 | 4 | 1 | 20.00% | rotosolve |
| `crlqas__L3_CH2_Singlet_8q__L3_CH2_Singlet_8q_rotosolve_s2_20k__seed11111__ep2199__snap0` | 0.000000 | 0.000000 | 0.000000 | 8 | 1 | 7 | 4 | 4 | 50.00% | rotosolve |
| `crlqas__L3_CH2_Singlet_8q__L3_CH2_Singlet_8q_rotosolve_s2_20k__seed11111__ep8360__snap0` | 0.000000 | 0.000000 | 0.000000 | 9 | 2 | 7 | 4 | 5 | 55.56% | rotosolve |
| `crlqas__L3_CH2_Singlet_8q__L3_CH2_Singlet_8q_rotosolve_s2_20k__seed11111__ep17692__snap0` | 0.000000 | 0.000000 | 0.000000 | 5 | 2 | 3 | 4 | 1 | 20.00% | rotosolve |
| `crlqas__L3_CH2_Singlet_8q__L3_CH2_Singlet_8q_rotosolve_s2_20k__seed11111__ep11867__snap0` | 0.000000 | 0.000000 | 0.000000 | 13 | 2 | 11 | 4 | 9 | 69.23% | rotosolve |
| `crlqas__L3_CH2_Singlet_8q__L3_CH2_Singlet_8q_rotosolve_s2_20k__seed11111__ep18342__snap0` | 0.000000 | 0.000000 | 0.000000 | 6 | 2 | 4 | 4 | 2 | 33.33% | rotosolve |
| `crlqas__L3_CH2_Singlet_8q__L3_CH2_Singlet_8q_rotosolve_s2_20k__seed11111__ep14078__snap0` | 0.000000 | 0.000000 | 0.000000 | 5 | 1 | 4 | 4 | 1 | 20.00% | rotosolve |
| `crlqas__L3_CH2_Singlet_8q__L3_CH2_Singlet_8q_rotosolve_s2_20k__seed11111__ep11050__snap0` | 0.000000 | 0.000000 | 0.000000 | 4 | 1 | 3 | 4 | 0 | 0.00% | rotosolve |
| `crlqas__L3_CH2_Singlet_8q__L3_CH2_Singlet_8q_rotosolve_s2_20k__seed11111__ep19196__snap0` | 0.000000 | 0.000000 | 0.000000 | 6 | 1 | 5 | 4 | 2 | 33.33% | rotosolve |
| `crlqas__L3_CH2_Singlet_8q__L3_CH2_Singlet_8q_rotosolve_s2_20k__seed11111__ep15155__snap0` | 0.000000 | 0.000000 | 0.000000 | 4 | 2 | 2 | 4 | 0 | 0.00% | rotosolve |
| `crlqas__L3_CH2_Singlet_8q__L3_CH2_Singlet_8q_rotosolve_s2_20k__seed11111__ep14648__snap0` | 0.000000 | 0.000000 | 0.000000 | 5 | 2 | 3 | 4 | 1 | 20.00% | rotosolve |
| `crlqas__L3_CH2_Singlet_8q__L3_CH2_Singlet_8q_rotosolve_s2_20k__seed11111__ep13334__snap0` | 0.000000 | 0.000000 | 0.000000 | 5 | 1 | 4 | 4 | 1 | 20.00% | rotosolve |
| `crlqas__L3_CH2_Singlet_8q__L3_CH2_Singlet_8q_rotosolve_s2_20k__seed11111__ep15771__snap0` | 0.000000 | 0.000000 | 0.000000 | 5 | 1 | 4 | 4 | 1 | 20.00% | rotosolve |
| `crlqas__L3_CH2_Singlet_8q__L3_CH2_Singlet_8q_rotosolve_s2_20k__seed11111__ep15997__snap0` | 0.000000 | 0.000000 | 0.000000 | 5 | 2 | 3 | 4 | 1 | 20.00% | rotosolve |
| `crlqas__L3_CH2_Singlet_8q__L3_CH2_Singlet_8q_rotosolve_s2_20k__seed11111__ep19064__snap0` | 0.000000 | 0.000000 | 0.000000 | 7 | 2 | 5 | 4 | 3 | 42.86% | rotosolve |
| `crlqas__L3_CH2_Singlet_8q__L3_CH2_Singlet_8q_rotosolve_s2_20k__seed11111__ep14541__snap0` | 0.000000 | 0.000000 | 0.000000 | 7 | 1 | 6 | 4 | 3 | 42.86% | rotosolve |
| `crlqas__L3_CH2_Singlet_8q__L3_CH2_Singlet_8q_rotosolve_s2_20k__seed11111__ep11060__snap0` | 0.000000 | 0.000000 | 0.000000 | 5 | 1 | 4 | 4 | 1 | 20.00% | rotosolve |
| `crlqas__L3_CH2_Singlet_8q__L3_CH2_Singlet_8q_rotosolve_s2_20k__seed11111__ep16699__snap0` | 0.000000 | 0.000000 | 0.000000 | 7 | 1 | 6 | 4 | 3 | 42.86% | rotosolve |
| `crlqas__L3_CH2_Singlet_8q__L3_CH2_Singlet_8q_rotosolve_s2_20k__seed11111__ep13586__snap0` | 0.000000 | 0.000000 | 0.000000 | 5 | 1 | 4 | 4 | 1 | 20.00% | rotosolve |
| `crlqas__L3_CH2_Singlet_8q__L3_CH2_Singlet_8q_rotosolve_s2_20k__seed11111__ep15909__snap0` | 0.000000 | 0.000000 | 0.000000 | 7 | 2 | 5 | 4 | 3 | 42.86% | rotosolve |
| `crlqas__L3_CH2_Singlet_8q__L3_CH2_Singlet_8q_rotosolve_s2_20k__seed11111__ep3356__snap0` | 0.000000 | 0.000000 | 0.000000 | 16 | 0 | 16 | 5 | 11 | 68.75% | rotosolve |
| `crlqas__L3_CH2_Singlet_8q__L3_CH2_Singlet_8q_rotosolve_s2_20k__seed11111__ep8273__snap0` | 0.000000 | 0.000000 | 0.000000 | 8 | 2 | 6 | 4 | 4 | 50.00% | rotosolve |
| `crlqas__L3_CH2_Singlet_8q__L3_CH2_Singlet_8q_rotosolve_s2_20k__seed11111__ep8238__snap0` | 0.000000 | 0.000000 | 0.000000 | 7 | 0 | 7 | 4 | 3 | 42.86% | rotosolve |
| `crlqas__L3_CH2_Singlet_8q__L3_CH2_Singlet_8q_rotosolve_s2_20k__seed11111__ep14134__snap0` | 0.000000 | 0.000000 | 0.000000 | 5 | 2 | 3 | 4 | 1 | 20.00% | rotosolve |
| `crlqas__L3_CH2_Singlet_8q__L3_CH2_Singlet_8q_rotosolve_s2_20k__seed11111__ep15881__snap0` | 0.000000 | 0.000000 | 0.000000 | 6 | 3 | 3 | 4 | 2 | 33.33% | rotosolve |
| `crlqas__L3_CH2_Singlet_8q__L3_CH2_Singlet_8q_rotosolve_s2_20k__seed11111__ep12876__snap0` | 0.000000 | 0.000000 | 0.000000 | 11 | 2 | 9 | 4 | 7 | 63.64% | rotosolve |
| `crlqas__L3_CH2_Singlet_8q__L3_CH2_Singlet_8q_rotosolve_s2_20k__seed11111__ep8193__snap0` | 0.000000 | 0.000000 | 0.000000 | 7 | 2 | 5 | 4 | 3 | 42.86% | rotosolve |
| `crlqas__L3_CH2_Singlet_8q__L3_CH2_Singlet_8q_rotosolve_s2_20k__seed11111__ep19889__snap0` | 0.000000 | 0.000000 | 0.000000 | 6 | 2 | 4 | 4 | 2 | 33.33% | rotosolve |
| `crlqas__L3_CH2_Singlet_8q__L3_CH2_Singlet_8q_rotosolve_s2_20k__seed11111__ep7970__snap0` | 0.000000 | 0.000000 | 0.000000 | 11 | 2 | 9 | 4 | 7 | 63.64% | rotosolve |
| `crlqas__L3_CH2_Singlet_8q__L3_CH2_Singlet_8q_rotosolve_s2_20k__seed11111__ep16972__snap0` | 0.000000 | 0.000000 | 0.000000 | 5 | 1 | 4 | 4 | 1 | 20.00% | rotosolve |
| `crlqas__L3_CH2_Singlet_8q__L3_CH2_Singlet_8q_rotosolve_s2_20k__seed11111__ep16349__snap0` | 0.000000 | 0.000000 | 0.000000 | 7 | 2 | 5 | 4 | 3 | 42.86% | rotosolve |
| `crlqas__L3_CH2_Singlet_8q__L3_CH2_Singlet_8q_rotosolve_s2_20k__seed11111__ep15799__snap0` | 0.000000 | 0.000000 | 0.000000 | 16 | 3 | 13 | 4 | 12 | 75.00% | rotosolve |

**Exact Retained-Structure Matches**

- count=33: `RY(q=1,θ=+0.180) | RY(q=5,θ=-3.142) | RX(q=7,θ=+3.142) | CNOT(1->5)`
- count=6: `RY(q=7,θ=+3.142) | RY(q=1,θ=+0.180) | RY(q=5,θ=-3.142) | CNOT(1->5)`
- count=6: `RY(q=1,θ=+0.180) | RY(q=7,θ=+3.142) | RY(q=5,θ=-3.142) | CNOT(1->5)`
- count=5: `RY(q=1,θ=+0.180) | RY(q=5,θ=-3.142) | RY(q=7,θ=+3.142) | CNOT(1->5)`
- count=4: `RY(q=5,θ=-3.142) | RY(q=1,θ=+0.180) | RX(q=7,θ=+3.142) | CNOT(1->5)`
- count=4: `RX(q=6,θ=+3.142) | RY(q=0,θ=+0.180) | RY(q=4,θ=-3.142) | CNOT(0->4)`
- count=4: `RY(q=1,θ=+0.180) | RY(q=5,θ=-3.142) | CNOT(5->7) | CNOT(1->5)`
- count=3: `RX(q=5,θ=-3.142) | RY(q=7,θ=+3.142) | RY(q=1,θ=+0.180) | CNOT(1->5)`
- count=3: `RY(q=5,θ=-3.142) | RY(q=1,θ=+0.180) | CNOT(5->7) | CNOT(1->5)`
- count=3: `RY(q=1,θ=+0.180) | RX(q=7,θ=+3.142) | RY(q=5,θ=-3.142) | CNOT(1->5)`

**Common Retained Gate Signatures**

- none
