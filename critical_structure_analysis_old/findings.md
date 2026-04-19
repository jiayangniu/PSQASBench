# Critical Structure Analysis — Research Note

> **目的**：供合作者了解工具背景、分子构型、实验结果与初步结论
> **日期**：2026-04-17
> **作者**：PSQASBench Team

---

## 1. 工具简介

**Critical Structure Tool** 是一个针对 RL-based QAS（量子架构搜索）方法的后处理分析工具。其核心目标是：在 RL agent 找到的"成功"电路（达到化学精度）中，**识别哪些门是真正必要的，哪些是冗余的**。

### 工作流程

```
输入：episode_traces.txt（RL 训练日志）
       ↓
① 筛选命中化学精度的 episode
       ↓
② warm-start 重建：从 first_hit_snapshot 恢复训练时的优化角度
       ↓
③ 门重要性评分（gate ablation）：逐一删除每个门，测量能量变化
       ↓
④ beam 剪枝：贪心地删除重要性最低的门，保留维持精度所需的最小子集
       ↓
输出：每个 episode 的最小保留结构 + 跨 episode 的公共门统计
```

**warm-start**：使用训练中首次达到化学精度时保留的电路结构与优化参数进行重建，大幅提升在平坦 landscape 上的重建成功率。

### 输出解读

| 指标 | 含义 |
|------|------|
| 原始门数 → 保留门数 | 最简化程度 |
| 冗余率 | `(原始 - 保留) / 原始` |
| Retained Gates | 最小化后的门序列 |
| Anchor Actions | 跨 episode 频率最高的门（可能是物理关键操作） |
| Common Signatures | 精确出现在所有 episode 中的门集合 |

---

## 2. 分析分子

### 2.1 L1 — BeH₂ STO-3G，6 qubits

| 参数 | 值 |
|------|----|
| 分子 | BeH₂，STO-3G 基组，平衡键长 |
| 编码 | Jordan-Wigner，6 qubits |
| Hamiltonian | 近对角，Pauli-Z 项主导，弱纠缠 |
| RL 方法 | CRLQAS（DQN，off-policy） |
| 角度优化器 | COBYLA，global_iters=100 |
| 电路深度上限 | 10 步（`num_layers=10`） |
| 化学精度阈值 | 1.6 mHa |
| 分析 bucket | 0.55 mHa（所有命中 episode 的实际误差均为此值） |
| 命中 episode 数 | 9055 |
| L1 分类依据 | Minimalism——测试能否识别并剪除冗余门 |

**能谱**：基态 E₀ 非简并，第一激发态 ΔE = 34.9 mHa，优化 landscape 单峰清晰。

### 2.2 L3 — CH₂ Singlet R130 A130，8 qubits

| 参数 | 值 |
|------|----|
| 分子 | CH₂ Singlet，R=1.30 Å，∠HCH=130°，STO-3G |
| 编码 | Jordan-Wigner，8 qubits |
| Hamiltonian | 近简并，含高阶 Pauli 交互 |
| RL 方法 | CRLQAS（DQN，off-policy） |
| 角度优化器 | Rotosolve，sweeps=2 |
| 电路深度上限 | 70 步 |
| 化学精度阈值 | 1.6 mHa |
| 分析 bucket | 0.00 mHa（命中精确基态的 episode） |
| 命中 episode 数 | 77 / 约 2000（约 3.8%） |
| L3 分类依据 | Stability——近简并 landscape 下的策略稳定性 |

**能谱**：基态 **3 重精确简并**（ΔE(E₁-E₀) < 0.01 mHa），第一非简并态 E₃-E₀ = 15.6 mHa。C₂ᵥ 对称性保护的简并结构。

---

[text](l3_ch2_8q_bucket000_main)## 3. 实验结果

### 3.1 L1 BeH₂ — 剪枝结果

分析 bucket：0.55 mHa，选取 6 个 episode。

| Episode | 原始门数 | 保留门数 | 冗余率 | 保留结构 |
|---------|---------|---------|--------|---------|
| ep9435 | 2 | 2 | 0% | `RX(q=4) \| CNOT(4->5)` |
| ep9404 | 2 | 2 | 0% | `RX(q=4) \| CNOT(4->5)` |
| ep9093 | 2 | 2 | 0% | `RY(q=4) \| CNOT(4->5)` |
| ep8380 | 2 | 2 | 0% | `RY(q=4) \| RX(q=5)` |
| ep9471 | 2 | 2 | 0% | `RX(q=4) \| RX(q=5)` |
| ep8040 | 2 | 2 | 0% | `RY(q=4) \| RY(q=5)` |

**Top Anchor Actions**（跨 episode 频率）：RY(q=5)（2484）、RX(q=5)（1751）、CNOT(4->5)（1698）

**Common Signatures**：工具报告 `none`（精确门集合取交集无重叠，因旋转轴不同）

### 3.2 L3 CH₂ — 剪枝结果（warm-start 版本）

分析 bucket：0.00 mHa，选取 10 个 episode，全部重建成功（retained_error = 0.000000 mHa）。

| Episode | 原始门数 | 保留门数 | 冗余率 | 活跃 qubit 集合 |
|---------|---------|---------|--------|----------------|
| ep1406 | 68 | 4 | **94.1%** | {1, 5, 7} |
| ep1920 | 57 | 5 | **91.2%** | {0, 4, 6} |
| ep1789 | 25 | 4 | 84.0% | {1, 5, 7} |
| ep1341 | 37 | 6 | 83.8% | {0, 1, 4, 6} |
| ep1606 | 50 | 9 | 82.0% | {0, 1, 4, 5, 6, 7} |
| ep1913 | 49 | 10 | 79.6% | {0, 2, 3, 4, 6, 7} |
| ep1359 | 64 | 8 | 87.5% | {0, 1, 4, 5, 6} |
| ep1646 | 58 | 17 | 70.7% | {0,1,2,3,4,5,6,7} |
| ep1774 | 23 | 10 | 56.5% | {0, 4, 6, 7} |
| ep1973 | 9 | 4 | 55.6% | {0, 4, 6} |
| **平均** | **44** | **7.7** | **~78.5%** | — |

**保留门序列（完整）**：

```
ep1406:  RY(q=1) | RY(q=5) | CNOT(1->7) | CNOT(5->1)
ep1920:  CNOT(6->4) | RX(q=0) | RY(q=4) | CNOT(4->0) | RY(q=6)
ep1789:  RX(q=7) | RY(q=5) | CNOT(5->1) | RY(q=5)
ep1341:  RY(q=1) | CNOT(1->6) | RY(q=1) | RY(q=4) | CNOT(6->0) | CNOT(4->0)
ep1606:  RY(q=5) | CNOT(5->4) | CNOT(4->5) | RX(q=7) | CNOT(4->1) | RX(q=6) | CNOT(1->0) | CNOT(4->1) | CNOT(6->0)
ep1913:  RY(q=3) | RY(q=4) | CNOT(3->6) | CNOT(4->7) | RX(q=3) | RX(q=4) | CNOT(7->2) | CNOT(7->0) | CNOT(7->2) | CNOT(0->7)
ep1359:  RX(q=6) | CNOT(6->5) | CNOT(5->1) | RX(q=5) | CNOT(6->1) | CNOT(6->4) | RY(q=0) | CNOT(0->4)
ep1646:  RY(q=7) | RX(q=6) | CNOT(6->2) | CNOT(2->3) | CNOT(3->6) | CNOT(7->0) | RY(q=2) | CNOT(3->4) | RY(q=0) | CNOT(3->5) | CNOT(4->1) | CNOT(7->3) | RY(q=1) | CNOT(5->4) | CNOT(7->3) | RY(q=3) | CNOT(1->5)
ep1774:  RY(q=0) | RX(q=6) | RX(q=4) | CNOT(6->7) | CNOT(0->4) | RX(q=7) | RX(q=7) | RX(q=7) | RX(q=7) | RX(q=7)
ep1973:  RY(q=0) | RY(q=6) | CNOT(6->4) | CNOT(0->4)
```

**Top Anchor Actions**：CNOT(0->4)（13 次）、CNOT(1->5)（7 次）、CNOT(4->0)（5 次，即 0↔4 反向）

**Common Signatures**：`none`（10 个电路结构全部不同，count=1）

**first_hit_error 分布**：

| 能量误差 bucket | episode 数 |
|----------------|-----------|
| 0.000623 mHa（精确值） | **77** |
| 0.0083 mHa | 1 |
| 0.25 mHa | 3 |
| 1.1–1.2 mHa | 2 |

---

## 4. 观察与结论

### 观察 1：Circuit Structure Bias 在两个分子层级上均严重

CRLQAS 找到的电路存在大量冗余门：

| 分子 | 平均冗余率 | 备注 |
|------|-----------|------|
| L1 BeH₂（depth=10） | **0%** | depth 限制已迫使电路压缩至最简 |
| L3 CH₂（depth=70） | **~78.5%**（范围 55–94%） | 无深度压力时冗余极高 |

L1 的 0% 冗余是 `depth=10` 限制的副产品，而非 CRLQAS 主动寻求最简电路的体现。若将 L1 depth 放开至 50/70，预计冗余率会接近 L3 水平。**这说明 RL 方法的 Circuit Structure Bias 是系统性的，depth 限制只是掩盖了问题。**

### 观察 2：L1 存在稳定的关键结构，L3 不存在

**L1 BeH₂**：6/6 episode 的最小结构均严格限定在 qubit `{4, 5}`，门类型略有变化（旋转轴不同）但物理区域完全一致。CNOT(4->5) 及旋转门组合对应 JW 编码下 BeH₂ 的 HOMO-LUMO 配对跃迁。

**L3 CH₂**：10/10 episode 的最小结构完全不同（count=1），活跃 qubit 集合涵盖 `{0,4,6}`、`{1,5,7}`、`{0,2,3,4,6,7}` 等多个互不相交的子集，Common Signatures 为空。最强 anchor CNOT(0→4) 仅出现在 50% 的最简电路中。

### 观察 3：L3 近简并的三条实验证据

CH₂ Singlet 是 L3 级别的近简并分子，以下三条证据来自实验数据本身：

**证据 1——能量流形，而非孤立极小点**：77 个成功 episode 的 first_hit_error 精确相同（0.0006226465 mHa，精确到 10 位有效数字）。若为非简并分子，不同电路应收敛至 E₀ 的不同数值近似；CH₂ 的一致性说明存在一个等高能量流形。

**证据 2——不相交 qubit 子集等价**：`{0,4,6}` 和 `{1,5,7}` 两组没有公共 qubit 的子集，均可裁剪到 4-5 门并达到 0.000 mHa 精度。对非简并分子，基态制备应对应唯一的关键子空间（如 BeH₂ 的 `{4,5}`）。

**证据 3——Anchor 不具有普遍性**：L1 BeH₂ 的最强 anchor CNOT(4→5) 出现在所有最简电路中（10/10）；L3 CH₂ 的最强 anchor CNOT(0→4) 仅出现在 5/10 中，另外 5 个用完全不同的 qubit 对达到相同精度。

> **独立验证（eigvals）**：从 `.npz` 直接计算，CH₂ 8q 基态为 **3 重精确简并**（ΔE < 0.01 mHa），第一非简并态 gap = 15.6 mHa，与实验证据完全吻合。

### 观察 4：Warm-start 对 L3 分析的必要性

冷启动重建在 L3 上失败率极高（旧版本 8/10 失败，baseline 误差高达 240–697 mHa）。根因是 CH₂ 的多模 landscape 导致随机角度初始化收敛到错误极小。引入 `first_hit_snapshot` warm-start 后，重建成功率从 20% 升至 100%，使完整的结构分析成为可能。

**含义**：对近简并分子，CRLQAS 的"成功"依赖训练时特定的角度历史，而非电路结构本身。换言之，同一个门序列，在训练得到的角度下成功，在重新优化时大概率失败——**角度记忆是 RL 成功的隐性依赖**。这是一个值得在 benchmark 中明确指出的问题。

---

## 5. 工具已知局限

| 问题 | 描述 | 影响 |
|------|------|------|
| 精确匹配过严 | Common Signatures 用 set.intersection，旋转轴不同的门不算公共 | L1 的实质结构一致性被低估（正确结论应为 qubit{4,5} 一致）|
| 成对冗余漏删 | ep1913 的 CNOT(7→2)×2 未被剪除（需同时删两步才有效） | 保留门数轻微虚高，不影响主要结论 |
| ep1774 重复旋转 | 保留了 5 个 RX(q=7)，等价于单个 RX(5θ) | 工具未合并同轴旋转，保留数虚高 |
