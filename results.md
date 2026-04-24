# 基准分层设计证据

本文档只保留支撑基准设计动机与 level 划分的结果，不是完整实验日志。

整体逻辑如下：

1. 先用 Pauli-string 与 Hamiltonian-matrix 诊断量，在任何 QAS 运行之前定义预期难点来源。
2. 训练后的结果只作为支撑证据：关键结构分析用来说明方法实际学到了什么电路家族，fidelity / best-error 结果用来验证每个 level 是否确实呈现出预期行为。

## 1. 用于 Level 设计的静态诊断

该基准使用如下从 qubit Hamiltonian 导出的静态描述符：

| 描述符 | 含义 |
|---|---|
| `Gap01 = E1 - E0` | 低能级间隔 / 近简并程度 |
| `Hub`, `Asym` | 相互作用集中度 / qubit 非对称性 |
| `Z-only`, `XY-only`, `Mixed`, `>=4-body` | Pauli-string 的局域性 / 关联负担 |
| `G1-G4` | 基于 Gershgorin 的矩阵诊断量，用于衡量对角占优 / 谱分离 |

这些静态诊断量在训练前定义各 level 的目标难点。搜索阶段的指标，如 best error 或 depth，只作为支撑性证据，而不是 level 划分的主依据。

### 1.1 主锚点与补充支撑案例

| Level | 角色 | 分子 | q | Gap01 | Hub | Asym | Z-only | XY-only | Mixed | >=4-body | G1 | G2 | 主要信号 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `L1` | 主案例 | `L1_BeH2_STO3G_6q` | 6 | 0.2120 | 1.1187 | 0.3560 | 0.9784 | 0.0216 | 0.0000 | 0.0216 | 0.9688 | 0.0937 | 极简、强局域、弱混合 |
| `L2` | 主案例 | `L2_LiH_Equil_6q` | 6 | 0.0772 | 1.1349 | 0.2556 | 0.8759 | 0.0648 | 0.0594 | 0.0847 | 0.8906 | 0.2249 | 中等非对称 / hub 偏置 |
| `L3` | 主案例 | `L3_CH2_Singlet_8q` | 8 | 0.0000 | 1.1616 | 0.3269 | 0.9067 | 0.0435 | 0.0499 | 0.0933 | 0.9453 | 0.2099 | 精确近简并，具有分支敏感性 |
| `L4` | 主案例 | `L4_H2_Stretch_4q` | 4 | 0.0044 | 1.0186 | 0.0371 | 0.7280 | 0.2720 | 0.0000 | 0.2720 | 0.7500 | 0.1800 | 小规模但表示负担重 |
| `L4` | 补充案例 | `L4_H2O_StrongCorr_8q` | 8 | 0.0939 | 1.3888 | 0.6568 | 0.7235 | 0.1011 | 0.1754 | 0.2168 | 0.6484 | 0.0738 | 更强相关负担与非对称性 |
| `L5` | 主案例 | `L5_H4_Chain_8q` | 8 | 0.2326 | 1.1188 | 0.3247 | 0.5989 | 0.1487 | 0.2524 | 0.4011 | 0.4180 | 0.0109 | 拓扑 / 路由压力 |
| `L5` | 补充案例 | `L5_H3_Linear_6q` | 6 | 0.0000 | 1.1462 | 0.2298 | 0.7091 | 0.1544 | 0.1366 | 0.2909 | 0.6562 | 0.0209 | 更小的拓扑敏感案例 |

### 1.2 L6 同分子家族的可扩展性阶梯

`L6` 不是单个孤立分子，而是 BeH2 同一家族上的扩展轴。

| 案例 | q | Gap01 | Hub | Asym | Z-only | XY-only | Mixed | >=4-body | G1 | G2 | BestDepth | Best Error (mHa) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `L6_BeH2_631G_8q` | 8 | 0.0895 | 1.1145 | 0.2283 | 0.9530 | 0.0470 | 0.0000 | 0.0470 | 0.9805 | 0.0462 | 50 | 0.116 |
| `L6_BeH2_6311G_10q` | 10 | 0.0627 | 1.0956 | 0.2531 | 0.9231 | 0.0349 | 0.0420 | 0.0769 | 0.9629 | 0.0105 | 45 | 0.248 |
| `L6_BeH2_CCPVDZ_12q` | 12 | 0.0677 | 1.1430 | 0.2538 | 0.8889 | 0.0303 | 0.0808 | 0.1111 | 0.9043 | 0.0192 | 63 | 1.243 |
| `L6_BeH2_CCPVDZ_14q` | 14 | 0.0615 | 1.1333 | 0.3765 | 0.7671 | 0.0520 | 0.1808 | 0.2329 | 0.6210 | 0.0003 | 91 | 5.008 |

解释：分子家族固定，但随着 basis 扩展，混合项与高阶项负担持续增长，当前搜索质量也从 `8q` 到 `14q` 明显退化。

## 2. 各 Level 的支撑性证据

### L1: BeH2 是极简性 / 深度敏感性锚点

| 视角 | `depth10` | `depth50` | 说明 |
|---|---|---|---|
| `best_eval` | [file](results/crlqas/L1_BeH2_STO3G_6q/Formal_EXP/L1_BeH2_STO3G_6q_cobyla_20k_depth10/seed11111/best_eval.txt): `0.554444 mHa`, depth `3`, CNOT `0`, `SR@chem = 0.9`, `mean_error_mha = 21.7348` | [file](results/crlqas/L1_BeH2_STO3G_6q/Formal_EXP/L1_BeH2_STO3G_6q_cobyla_20k_depth50/seed11111/best_eval.txt): `0.554444 mHa`, depth `5`, CNOT `2`, `SR@chem = 1.0`, `mean_error_mha = 0.5545` | 稳定的策略输出对应的是浅层 `~0.55 mHa` 家族，而不是更低误差家族。 |
| `best_train` | [file](results/crlqas/L1_BeH2_STO3G_6q/Formal_EXP/L1_BeH2_STO3G_6q_cobyla_20k_depth10/seed11111/best_train.txt): ep `167`, `0.268226 mHa`, depth `9`, CNOT `4` | [file](results/crlqas/L1_BeH2_STO3G_6q/Formal_EXP/L1_BeH2_STO3G_6q_cobyla_20k_depth50/seed11111/best_train.txt): ep `582`, `0.268226 mHa`, depth `9`, CNOT `5` | 两种深度预算最终都能发现同一个更低误差、较紧凑的家族。 |

对 `depth50` 而言，全局最优轨迹是先用明显更深的电路进入低误差区间，之后才塌缩到相同的 depth-`9` 紧凑家族。这更像是搜索过程中的结构压缩，而不是简单的“depth 越大越好”；见 [episode_summary.tsv](results/crlqas/L1_BeH2_STO3G_6q/Formal_EXP/L1_BeH2_STO3G_6q_cobyla_20k_depth50/seed11111/episode_summary.tsv)。

关键结构分析进一步明确了这种分裂。在 `~0.55 mHa` 档，两个 depth 设置都会剪枝到同一个极小的 `q4/q5` 结构家族；见 [depth10 0.55](critical_structure_analysis/l1_beh2_depth10_bucket055/summary.md) 与 [depth50 0.55](critical_structure_analysis/l1_beh2_depth50_bucket055/summary.md)。

在 `~0.27 mHa` 档，保留下来的精确结构签名全部唯一，而且两个 bucket 都没有公共的保留门签名；见 [depth10 0.27](critical_structure_analysis/l1_beh2_depth10_bucket027/summary.md) 与 [depth50 0.27](critical_structure_analysis/l1_beh2_depth50_bucket027/summary.md)。但两个 fidelity 分析仍然都塌缩到同一个重新优化后的状态簇，且簇内最小 fidelity 为 `0.9992`；见 [depth10 fidelity](critical_structure_analysis/l1_beh2_depth10_bucket027/fidelity_analysis.md) 与 [depth50 fidelity](critical_structure_analysis/l1_beh2_depth50_bucket027/fidelity_analysis.md)。同时，`depth50` bucket 保留下来的剪枝骨架更大，最多仍有 `14` 个保留门，而 `depth10` 只剩 `5-6` 个，这与更深训练轨迹会学出更强耦合、也更难被干净剪枝的中间结构是一致的。这个结果支持 RLQAS 采用按分子选择 depth，或引入自适应 depth 机制。

### L2: LiH 是非对称性 / 相互作用枢纽锚点

- 相比 `L1`，`L2_LiH_Equil_6q` 具有更高的 hub 与非对称压力：`Hub = 1.1349`, `Asym = 0.2556`。
- 它的 Pauli 组成也不再像 `L1` 那样纯局域：`Mixed = 0.0594`, `>=4-body = 0.0847`，都明显高于 `L1`。
- Qubit 重要性分布：`[0.5283, 0.5607, 0.6818, 0.6084, 0.6127, 0.6127]`，**真正的 hub 是 q=2（0.6818）**，q=4/q=5 是次要节点（各 0.6127）。

**观察到的支撑证据：**

**现象：agent 识别了错误的 hub（q=4/q=5 而非 q=2）**

CRLQAS 在 LiH 6q 上展现出典型的错误 hub 偏置行为：

- **1.05 mHa 平台期（化学精度附近）**：训练全程有 `15862` 个 snapshot 事件落在此 bucket，占绝对多数。对这些 episode 剪枝后，`9/10` 的保留结构都压缩到仅 `2` 个门，另有 `1/10` 保留 `3` 个门；主导 motif 几乎全部集中在 `q=4/q=5`。anchor actions 也完全由 `RX(q=5)`（3022 次）、`RX(q=4)`（2718 次）、`RY(q=4)`（2195 次）主导，q=2 完全缺席；见 [bucket 1.05 mHa](critical_structure_analysis/l2_lih_6q_curriculum_bucket1p05/summary.md)。

- **0.5 mHa bucket（更优解，低于 1.05 mHa）**：当前分析下只观察到 `22` 个 snapshot 事件，agent 进入这一更优区域的频率明显更低。对这些 episode 剪枝后，所有 `10` 条保留结构均涉及 q=2；在 `18` 个 distinct anchor-action token 中，有 `7` 个 CNOT token 直接涉及 q=2（`CNOT(3->2)`×2、`CNOT(1->2)`、`CNOT(5->2)`、`CNOT(4->2)`、`CNOT(2->0)`、`CNOT(2->3)`、`CNOT(2->5)`；若计入 `RZ(q=2)` 则为 `8` 个）；保留电路规模也从 `1.05 mHa` 平台期的 `2-3` 个门跃升至 `10-24` 个门；见 [bucket 0.5 mHa](critical_structure_analysis/l2_lih_6q_curriculum_bucket0p5/summary.md)。

- **last_action 列的直接证据**：0.5 mHa bucket 中 10 个 episode 的最后一步动作，有 6 条直接涉及 q=2（`CNOT(2->5)`, `CNOT(3->2)`×2, `CNOT(1->2)`, `CNOT(2->3)`, `CNOT(4->2)`）。

**机制解读：**

一种与当前证据一致的解释是：agent 迅速学到 `CNOT(4->5)` / `RX(q=4)` / `RX(q=5)` 可以在 `~1.05 mHa` 处快速获得正 reward（课程阈值），于是 policy 优先收敛到 q=4/q=5 的局部最优。q=4/q=5 虽是次要 hub（重要性 0.6127），但两者相邻且重要性对称，提供了低成本的早期 reward 信号。真正的 hub q=2（重要性 0.6818）位于链中间；从 `0.5 mHa` bucket 中大量涉及 `q=2` 的 retained structures、anchor actions 与 last actions 来看，更优解往往需要显式调动这一中间节点，因此探索门槛也更高。

**结论：**

`L2` 成功暴露了 CRLQAS 在非对称 Hamiltonian 上的结构偏置：方法不只是”用了太多门”，而是**聚焦于错误的 qubit 子空间**。这是一个比 L1 的”电路冗余”更深层的失效模式，直接支持 `L2 Asymmetry` 作为独立诊断 level 的设计动机。

### L3: CH2 8q 是近简并 / 稳定性锚点

- `L3_CH2_Singlet_8q` 是 benchmark 中精确近简并程度最高的分子：`Gap01 = 0.0000`（E1 = E0，精确到数值精度）。
- Pauli 复杂度中等：`Mixed = 0.0499`, `>=4-body = 0.0933`, `Asym = 0.3269`。

**观察到的支撑证据：**

**CRLQAS 能找到 CH2 8q 的基态，但搜索路径极度不稳定**

- **成功率**：在 `0.00 mHa` bucket 中共有 `77` 个 episode 达到化学精度（`first_hit_error = 0.0006 mHa`），说明 CRLQAS 确实能够到达基态，不是简单的”无法求解”。
- **电路深度的极端差异**：10 个抽样 episode 的首次成功步数为 `9`、`23`、`64`、`50`、`57`、`68`、`49`、`25`、`58`、`37`——最短 `9` 步，最长 `68` 步，差异高达 7 倍。
- **剪枝后门数范围 4–17**：剪枝至 `0.0000 mHa` 后保留门数为 `4`（ep1973）到 `17`（ep1646），原始电路的冗余度为 `55.56%` 到 `94.12%`。
- **无公共保留结构**：10 条保留结构的精确签名全部唯一（每个 `count = 1`），连最基本的 2-gate 核心也不共享；见 [summary.md](critical_structure_analysis/l3_ch2_8q_bucket000_main/summary.md)。
- **anchor actions 分散**：`CNOT(0->4)` 出现 `13` 次，是最主要的 anchor，但排名第二的 `CNOT(1->5)` 只有 `7` 次，第三的 `CNOT(4->0)` 只有 `5` 次，其余 `33` 个 token 各出现 `1–3` 次。没有单一”必经之路”。

**核心发现：真实物理简并导致两个正交基态**

fidelity 分析揭示了这种结构分散背后的物理根因。10 条保留结构在独立重新优化后形成 **2 个确定性 complete-linkage 簇**，且跨簇 fidelity 精确为 `0.0000`（两个量子态完全正交）；见 [fidelity_analysis.md](critical_structure_analysis/l3_ch2_8q_bucket000_main/fidelity_analysis.md)：

| 簇 | 成员数 | 重优化能量误差 | 代表结构 |
|---|---:|---:|---|
| 簇 0 | 7 | 0.0000 mHa | `RY(q=0) \| RY(q=6) \| CNOT(6->4) \| CNOT(0->4)` 等，均涉及 q=0/4/6 |
| 簇 1 | 3 | 0.0000 mHa | `RY(q=1) \| RY(q=5) \| CNOT(1->7) \| CNOT(5->1)` 等，均涉及 q=1/5/7 |

两个簇都能达到 `0.0000 mHa`，说明两者均为真实基态而非次优解。跨簇 fidelity = `0.0000` 表明这是两个能量简并但量子态正交的基态——CH2 Singlet 在 JW 编码下由 α/β 自旋轨道交换对称性产生的二重简并。簇 0 的结构集中在偶数轨道（q=0,4,6），簇 1 集中在奇数轨道（q=1,5,7），与 α/β 轨道 interleaving 的 JW 编码方式完全吻合。


### L4: 表示 / 关联负担由一个小锚点和一个更大补充案例共同支撑

- `L4_H2_Stretch_4q` 被选为主小锚点，因为它虽然只有 `4` qubit，但已经表现出明显的表示负担：`Gap01 = 0.0044`, `XY-only = 0.2720`, `>=4-body = 0.2720`, `G1 = 0.7500`, `G2 = 0.1800`。
- `L4_H2O_StrongCorr_8q` 被保留为更大的相关补充案例，因为它在更高负担下体现了相同 level 的含义：`Hub = 1.3888`, `Asym = 0.6568`, `Mixed = 0.1754`, `>=4-body = 0.2168`, `G2 = 0.0738`。

观察到的支撑证据：

- 已存档的 `H2O` sub-`10 mHa` 分布并未达到 chemical accuracy，而是集中在一个较窄的退化误差区间。在 [bucket_summary.tsv](critical_structure_analysis/l4_h2o_sub10mha_list/bucket_summary.tsv) 中，`67` 个事件全部落在 `5.88` 到 `7.33 mHa` 之间，其中主桶是 `6.46 mHa`，包含 `52` 个事件。
- 这与 `L4` 的设计意图一致：这里的难点不仅仅是小能隙，而是即便在中等规模下仍然显著的表示 / 关联负担。

结论：

- `L4` 应被描述为表示 / 关联 level，其中 `H2_Stretch` 是干净的小锚点，拉伸 `H2O` 是更大的支撑案例。

### L5: 拓扑 / 路由压力应在多种 CNOT 连接性下测试

- `L5_H4_Chain_8q` 是主拓扑锚点，因为它在已选主 levels 中具有最强的非局域 Pauli 负担：`Z-only = 0.5989`, `Mixed = 0.2524`, `>=4-body = 0.4011`, `G1 = 0.4180`, `G2 = 0.0109`。
- `L5_H3_Linear_6q` 是更小的补充拓扑案例：`Gap01 = 0.0000`, `Mixed = 0.1366`, `>=4-body = 0.2909`, `G2 = 0.0209`。

观察到的支撑证据：

- 在当前 `all-connectivity` 的 `2.0 mHa` bucket 中，[summary.md](critical_structure_analysis/l5_h3_linear_topology_all_bucket2p0/summary.md) 显示保留结构不会塌缩成一个精确电路，而是存在 `2` 个精确保留骨架（各 `count=2`）。
- 这两个家族仍共享一组非局域 CNOT 核心：`CNOT(0->1)`, `CNOT(0->5)`, `CNOT(3->0)`, `CNOT(3->4)`, `CNOT(3->5)`, `CNOT(5->2)`。
- 保留电路也仍然偏大：原始 `43-47` 个 gates 只能剪到 `29-33` 个保留门，冗余度约 `30%`。

结论：

- `L5` 不应被视为单一固定化学实例。它本质上是一个拓扑 benchmark，因此应在多种 CNOT 连接性（`all` 与 `linear`）下运行。相应的 `Topology_EXP` 配置现在已在 `crlqas` 与 `hyrlqas` 中提供。

### L6: 同家族 scaling 才是正确的“最高难度 level”叙事

- BeH2 阶梯在保持分子家族不变的前提下提升 basis / active-space 负担。
- 从 `8q` 到 `14q`，Pauli 结构逐渐变得不再局域：`Z-only: 0.9530 -> 0.7671`, `Mixed: 0.0000 -> 0.1808`, `>=4-body: 0.0470 -> 0.2329`；矩阵诊断量也明显变弱：`G1: 0.9805 -> 0.6210`, `G2: 0.0462 -> 0.0003`。

观察到的支撑证据：

- 当前最优运行统计已经体现出同家族 scaling 趋势：best error 从 `0.116 mHa`（`8q`）上升到 `0.248 mHa`（`10q`）、`1.243 mHa`（`12q`）以及 `5.008 mHa`（可选扩展 `14q`）。
- Best depth 也整体上升，在 `14q` 达到 `91`。

结论：

- `L6` 不是“某一个最难分子”。它是在同一分子家族上构造出的可扩展性阶梯，这才是它在 benchmark 中应承担的角色。
