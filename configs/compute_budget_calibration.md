# Benchmark 计算预算校准说明

## 一、当前口径

这份说明记录 2026-04-27 这一轮 benchmark 预算调整后的统一理解。

当前不再强行把四种方法压到同一个“episode”定义，而是采用下面的原则：

- `RLQAS / TFQAS / GQEQAS` 尽量用“量子态相关计算量”来比较训练预算。
- `QuantumDARTS (qdarts)` 不能自然地等价成离散电路的 H-oracle 次数，需要单独拆成
  `soft-search budget` 和 `discrete-eval budget` 两部分报告。
- 目标不是精确相等，而是让不同方法对同一个分子的训练预算处于同一数量级，并保留
  RLQAS 可以更高一些这一事实。

本轮 `Formal_EXP` 配置统一采用下面的缩放规则：

| 方法 | 当前缩放规则 | 实际改动参数 |
|------|-------------|-------------|
| `CRLQAS` | 缩小到原来的 `0.5x` | `episodes` |
| `QDARTS` | 缩小到原来的 `0.5x` | `n_epochs` |
| `GQEQAS` | 扩大到原来的 `1.5x` | `epochs`, `warmup_epochs` |
| `TFQAS` | 扩大到原来的 `1.5x` | `S`, `R` |

同时保留此前已经确定的两条口径：

- `TFQAS` 的 `n_expr_samples` 固定为 `2000`
- `TFQAS / GQEQAS / QDARTS / RLQAS` 的最终评估候选规模保持各自当前设定，不在本轮一起改动

---

## 二、各方法更合理的预算模型

### 2.1 RLQAS

`RLQAS` 每一步都要在当前离散电路上调用外部优化器，然后再得到这一步的能量结果。

近似记法：

```text
B_train_RL ≈ sum_over_steps (nfev_step + 1)
B_select_RL ≈ N_eval_ckpt × eval_K × C_eval
```

其中：

- `nfev_step` 由 `COBYLA / Rotosolve / SPSA / PSRAdam` 决定
- episode 是否会提前终止，直接影响总步数
- curriculum、问题难度、深度上限都会改变真实预算

所以：

- `episodes` 只能作为训练长度参数，不能直接当预算
- 对 `RLQAS` 最稳妥的表述应该是“`episodes` 决定上限，真实预算由 step 数和每步优化器成本共同决定”

### 2.2 TFQAS

`TFQAS` 真正重的不是 Stage 1，而是 Stage 2 的 expressibility 估计。

更合理的近似记法：

```text
B_train_TF ≈ 2 × R × n_expr_samples
B_select_TF ≈ K × C_eval
```

解释：

- Stage 1 的 `path_count` 是结构代理，基本可视为零成本
- Stage 2 对每个候选电路要生成 `2 × n_expr_samples` 个 statevector
- Stage 3 只对 top-`K` 电路做离散优化，属于较小的选择成本

因此：

- `TFQAS` 的主预算应该由 `R` 和 `n_expr_samples` 决定
- 在 `n_expr_samples = 2000` 固定后，本轮选择放大 `S` 和 `R`

### 2.3 GQEQAS

`GQEQAS` 的 online 模式不是“每个 epoch 都重算一批电路”，而是每隔
`online_refresh_every` 个 epoch 才采样 fresh circuits。

更合理的记法应先从代码里的每个 refresh 周期写起：

```text
N_refresh ≈ epochs / online_refresh_every

target_batch  = online_sample_count
target_replay = round(target_batch × replay_mix_ratio)
replay_n      = min(target_replay, replay_available)
fresh_n       = max(1, target_batch - replay_n)

N_fresh_optimized = sum_over_refresh fresh_n

B_train_GQE ≈ sum_over_fresh_sequences (nfev_seq + prefix_len)
B_select_GQE ≈ N_eval_ckpt × eval_n_sequences × (nfev_seq + prefix_len)
```

解释：

- `training_reopt = 1` 时，只有 `fresh_n` 条 fresh sequence 会被外部优化器重优化
- 同时，训练目标还需要每个 token 前缀的能量，因此不能只数最终能量
- `replay_mix_ratio` 会通过 `fresh_n = target_batch - replay_n` 直接减少每个 refresh 周期
  被新采样并优化的电路数量

在 replay buffer 很快非空、并且 `replay_mix_ratio = r` 基本稳定生效时，可再写成更直观的近似：

```text
N_fresh_optimized ≈ online_sample_count + (N_refresh - 1) × online_sample_count × (1 - r)
```

因此：

- `GQEQAS` 的预算主要由 `epochs / refresh_every / online_sample_count / replay_mix_ratio`
  决定
- 本轮将 `epochs` 与 `warmup_epochs` 一起放大到 `1.5x`，保持 warmup 比例不变

### 2.4 QDARTS

`QDARTS` 是最难与其他方法一维对齐的方法，因为它训练时优化的是 soft-relaxed 超位置电路，
不是一条条离散电路。

更合理的记法应该拆成两部分：

```text
B_soft_qdarts = n_epochs × (n_inner + 1)
B_disc_qdarts = N_eval_ckpt × eval_k

B_soft_proxy ≈ B_soft_qdarts × n_slots × avg_candidate_count
```

解释：

- 每个 epoch 有 `n_inner` 次 `theta` 更新和 `1` 次结构更新
- 但一次 soft forward 不是“一条离散电路的一次 H 求值”
- 在 soft propagation 中，每个 slot 都会遍历该 qubit 上所有候选门，再把结果加权叠加

因此：

- `QDARTS` 不能直接说“等价于多少次 RLQAS episode”
- 更合理的做法是同时报告：
  - `soft-search passes`
  - `discrete eval circuits`
  - 如果需要，再额外给出 `branch-application proxy`

---

## 三、两个代表分子的当前预算量级

下面用两个代表分子来判断“当前设置是否处于合理量级”：

- `6q`: `L2_LiH_Equil_6q`
- `8q`: `L3_CH2_Singlet_8q`

这些数字是为了看量级，不追求伪精确。

### 3.1 `L2_LiH_Equil_6q`

当前仓库配置（以 `configs/...` 文件为准，不以已启动任务的历史日志为准）：

- `CRLQAS`: `episodes = 10000`
- `TFQAS`: `S = 22500`, `R = 2250`, `n_expr_samples = 2000`
- `GQEQAS`: `epochs = 6000`, `warmup_epochs = 600`, `online_sample_count = 250`, `online_refresh_every = 25`
- `QDARTS`: `n_epochs = 1600`, `n_inner = 50`, `eval_every = 80`, `eval_k = 30`

粗略预算：

| 方法 | 当前训练预算量级 | 备注 |
|------|----------------|------|
| `CRLQAS` | `O(10^7)` 到数倍 `10^7` | 取决于 early stop 和每步 COBYLA `nfev` |
| `TFQAS` | `2 × 2250 × 2000 = 9.0M` | 这部分是 Stage 2 statevector 预算 |
| `GQEQAS` | 约 `2M - 3M` | replay 会明显降低 fresh sequence 数 |
| `QDARTS` | `81600` 次 soft pass | 若乘 `20 × 6 × 7` 候选分支 proxy，约 `68.5M` |

判断：

- `TFQAS` 现在是合理的，已经进入和 RLQAS 同数量级的区间
- `GQEQAS` 仍然偏低，但方向上比之前更接近 benchmark 主体
- `QDARTS` 即便把 `n_epochs` 砍半，soft-search 代理预算依然不小，只是语义上不能和其他方法直接等价

### 3.2 `L3_CH2_Singlet_8q`

当前仓库配置（以 `configs/...` 文件为准，不以已启动任务的历史日志为准）：

- `CRLQAS`: `episodes = 10000`
- `TFQAS`: `S = 22500`, `R = 2250`, `n_expr_samples = 2000`
- `GQEQAS`: `epochs = 6000`, `warmup_epochs = 600`, `online_sample_count = 250`, `online_refresh_every = 25`
- `QDARTS`: `n_epochs = 2400`, `n_inner = 20`, `eval_every = 120`, `eval_k = 24`

粗略预算：

| 方法 | 当前训练预算量级 | 备注 |
|------|----------------|------|
| `CRLQAS` | 数倍 `10^7` | `Rotosolve s2` 的每步成本明显高于 6q COBYLA |
| `TFQAS` | `2 × 2250 × 2000 = 9.0M` | Stage 2 仍是主成本 |
| `GQEQAS` | 约 `6M - 8M` | 受平均 token 长度和 Rotosolve 参数数影响 |
| `QDARTS` | `50400` 次 soft pass | 若乘 `30 × 8 × 9` 候选分支 proxy，约 `108.9M` |

判断：

- `TFQAS` 和 `GQEQAS` 已经在相近数量级
- `CRLQAS` 依然更贵，但这符合“RLQAS 允许更高预算”的前提
- `QDARTS` 依旧是最难直接对齐的方法；当前最合理的做法不是继续强凑，而是单独报告 `soft-search` 成本

---

## 四、对旧版文档的评价

旧版文档不是完全没道理，但它更适合作为“第一次校准的草稿”，不适合继续当最终口径。

### 4.1 有道理的地方

- 它抓住了最重要的一点：不能用原始 `episodes` 或 `epochs` 直接比较四个方法
- 它意识到了 `TFQAS / GQEQAS / QDARTS` 的训练语义彼此不同
- 它已经开始把 `QDARTS` 看成一种与 RL 离散搜索不同的 oracle 使用方式，这个方向是对的

### 4.2 需要修正的地方

- 它把 `TFQAS Stage 1` 写成了主要 H-oracle 预算，这一点不对；真正重的是 Stage 2 expressibility
- 它把 `RLQAS` 的 `T_avg`、`C_avg` 写得过于固定，容易给人“预算已被精确标定”的错觉
- 它把 `QDARTS overhead_factor` 直接写成预算，这更像 proxy，而不是可与离散 H-oracle 等价的真实计数
- 它低估了 `GQEQAS` 的 prefix-energy 成本，同时也没有把 replay 对 fresh budget 的削减写清楚
- 它记录的很多具体配置数值已经与当前 `Formal_EXP` 不一致
- 它写到 “GQE 无 L5 Formal_EXP 配置” 也已经过时；现在 `L5_H3_Linear_6q` 和 `L5_H4_Chain_8q`
  的 `GQEQAS` 配置都已补上

---

## 五、当前更推荐的报告方式

正式 benchmark 建议同时报告下面三类信息，而不是试图压缩成单一“episode”：

| 指标 | 适用方法 | 含义 |
|------|---------|------|
| `B_train` | `RLQAS / TFQAS / GQEQAS` | 训练阶段量子态相关计算量 |
| `B_select` | 全部方法 | checkpoint / top-K / 最终离散评估成本 |
| `B_soft` | `QDARTS` | soft-relaxed 搜索主循环成本 |

如果表格只能放一个预算指标，那么优先顺序建议是：

1. `wall-clock`
2. `B_train` 或 `B_soft`
3. `B_select`

而不是直接放“`episodes` / `epochs` / `S`”。

---

## 六、当前结论

这一轮调整后的总体判断是：

- `TFQAS`：当前口径合理
- `GQEQAS`：比之前更接近目标，但 6q 仍偏保守
- `CRLQAS`：仍然比非RL方法更贵，但这是允许且可解释的
- `QDARTS`：不适合继续强行压成与其他三者同一个一维预算；应单独报告 soft-search 预算

换句话说，旧版文档“想做预算校准”这件事本身是对的，但它里面不少具体公式和结论现在已经不够稳了。
这份新版本保留了那个方向，同时把口径改成了更接近当前代码实现和配置状态的版本。
