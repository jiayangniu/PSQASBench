# PSQASBench — Paper Structure & Design Decisions

> 本文件记录论文结构讨论与关键设计决策。
> 🔴 标注表示**悬而未决**的问题，不可在实验/写作中预设结论。

---

## Motivation

**RL for QAS 的 evaluation 是 broken 的。**

现有方法在三个维度上存在系统性 inconsistency，导致方法间比较不可信：

| 维度 | 问题描述 |
|------|---------|
| **What to test on** | 各方法使用不同分子，无公共基准，结果无法横向比较 |
| **When to evaluate** | Checkpoint 策略混乱（last / best_energy / 不说明），结果不可复现 |
| **What to measure** | Energy error 单指标无法区分电路质量（2个CNOT达精度 vs 10个CNOT达精度） |

**本工作的回答**：提出 PSQASBench，系统性解决这三个问题。

---

## 贡献列表

1. **6-Tier Molecular Diagnostic Suite**：专为诊断 RL-QAS 失败模式设计的标准化分子集
2. **Principled Evaluation Protocol**：统一的实验设置与 checkpoint 策略（🔴 见下）
3. **新评估指标**：Pareto 视角 + Policy Circuit Diversity（PCD: D_struct / D_func）
4. **Benchmark Findings**：系统性暴露现有方法的缺陷
5. **CRLQAS-STOP**：概念验证，证明 Circuit Structure Bias 可修复

---

## 论文结构

### Section 1: Introduction
- RL for QAS 的研究背景与重要性
- Evaluation inconsistency 的三个维度（上表）
- 本工作贡献概述

### Section 2: Related Work
- RL for QAS 方法综述，重点展示各方法评估设置的差异（为 motivation 提供文献证据）
- 相邻领域 benchmark 工作对比

### Section 3: PSQASBench — A Principled Evaluation Framework

#### 3.1 标准化诊断数据集（解决"测什么"）

6-Tier Molecular Diagnostic Suite：

| Tier | 硬度来源 | 分子 | 诊断目标 |
|------|---------|------|---------|
| L1 | 基础优化 | H2(Equil.), BH | Minimalism：能否剪除冗余门 |
| L2 | 非对称/Interaction Hub | BeH⁺, LiH(Equil.), BF | Asymmetry：资源分配到 interaction hub |
| L3 | 近简并（小 Gap） | HeH⁺, CH₂, LiH(Stretch), H3(Triangle) | Stability：平坦 landscape 中的策略稳定性 |
| L4 | 强关联 | H2(Stretch), H3(Linear), H2O | Representation：高阶 Pauli 项的表达能力 |
| L5 | 拓扑路由 | H3(Linear), H4(Chain) | Topology：1D 连通性约束下的电路设计 |
| L6 | 规模化 | BeH2 | Scalability：action space 指数增长时的收敛效率 |

**Hamiltonian Fingerprint 指标**（用于描述分子结构特征，独立于方法结果）：

| 指标 | 定义 | 与失败模式的预期关联 |
|------|------|-------------------|
| Z-only 占比 | Z-only Pauli 项权重比 | 占比高 → 好电路 CNOT 数应少；方法若用很多 CNOT 则 Circuit Structure Bias 严重 |
| High-order ratio | ≥4-body Pauli 项权重占比 | 占比高 → 需要深纠缠结构；方法 SR@chem 预期更低 |
| Energy Gap ΔE | E1 - E0 | 接近 0 → landscape 平坦 → D_func 预期偏高 |
| Hub Score | max(Iq) / mean(Iq) | 🔴 与方法表现的具体关联需实验后再讨论，不预设 |
| Asymmetry | (max-min) / mean of Iq | 🔴 与方法表现的具体关联需实验后再讨论，不预设 |

> ⚠️ **注意**：Fingerprint 与方法表现的相关性分析是**数据驱动**的。
> 只有 Z-only 占比、High-order ratio、ΔE 有较强的理论依据。
> Hub Score 和 Asymmetry 与具体指标的对应关系**必须等实验数据出来后再讨论**，不可提前断言。

**关于 physical sector 的定位**：

- **主 benchmark 任务定义保持 full-space**。原因是当前 RLQAS/CRLQAS 等方法实际优化的就是 full Hamiltonian，对方法行为与失败模式的解释应优先对应它们真正面对的目标。
- **补充加入 physical-sector fingerprint analysis**，作为 benchmark validity / chemistry sanity-check，而不是替代主 benchmark 设定。
- 具体来说：
  - `full-space fingerprints`：解释当前方法实际上在解什么问题、为什么会难；
  - `physical-sector fingerprints`：检查这些难点中哪些反映分子的真实化学结构，哪些可能来自未限制 charge / spin sector 的 formulation artifact。
- 因此，physical sector 的作用不是“改写 benchmark 主任务”，而是**提高 benchmark 对分子难度解释的物理可信度**。

> 建议写法：正文中以 full-space 结果为主；physical-sector 分析作为一节补充诊断，专门讨论 tier 设计的化学有效性与可能的 sector-induced artifact。

#### 3.2 评估协议（解决"何时评估"）

**文献现状**（benchmark finding 的一部分）：

| 做法 | 代表工作 | 缺陷 |
|------|---------|------|
| 保存 last checkpoint | 多数论文 | 训练末期可能退化 |
| 保存 best energy checkpoint | CRLQAS、HyRLQAS | 偏向深电路 |
| 保存 best SR checkpoint | 极少数 | 忽略电路质量 |
| 不说明 | 大量论文 | 不可复现 |
| Checkpoint 完全禁用 | bench-rlqas (PPO/A2C) | 无法评估 policy |

**🔴 本 benchmark 的统一 checkpoint 方案：悬而未决**

候选方案：
- **方案A**：last checkpoint（最可复现，但不确定训练后期是否退化）
- **方案B**：训练全程周期性评估（每 N episode greedy rollout × 20次），取 SR@chem 最高的 checkpoint
- **方案C**：仅记录训练过程中的 global_best_energy，不依赖 checkpoint（只报告训练过程指标，不评估 policy）

> 该决策影响 PCD 的计算和 Pareto 分析。需要先跑少量实验，观察各方法训练曲线后再定。

**固定设置（已确定）**：
- 每个实验：≥5 个随机 seed
- 化学精度阈值：1.6 mHa
- Hamiltonian 编码：Jordan-Wigner
- 噪声模型：Noiseless only

#### 3.3 评估指标（解决"测什么指标"）

**主指标：Pareto 视角**
- 横轴：CNOT 门数（circuit cost）
- 纵轴：Energy Error（mHa）
- 目标：相同 circuit cost 下 energy error 最小

**辅助指标**：

| 指标 | 定义 |
|------|------|
| SR@chem | 多 seed 中达到化学精度的比例 |
| CNOT@chem | 首次达到化学精度时的 CNOT 数 |
| nfev@chem | 首次达到化学精度时的 VQE 函数评估次数 |
| Best energy error | 全程最低 energy error（不限深度） |

**新指标：Policy Circuit Diversity（PCD）**

给定固定 policy，K 个随机 seed 下生成 K 条电路：

- **D_struct**（结构多样性）：固定角度 θ=π/4，计算 CNOT 骨架的 pairwise HS 距离
- **x z**（功能多样性）：各电路独立优化至 θ*，计算优化后 Unitary 的 pairwise HS 距离

诊断矩阵：

| D_struct | D_func | 解读 |
|----------|--------|------|
| 低 | 低 | 理想：结构一致，优化收敛稳定 |
| 高 | 低 | 可接受：多等价电路，功能一致（Hamiltonian 对称性） |
| 低 | 高 | 结构统一但优化不稳定，landscape 问题 |
| 高 | 高 | 方法不可靠，随机游走 |

> 🔴 **PCD 的计算依赖 checkpoint 方案的确定**。checkpoint 方案未定前，PCD 的实验设计暂缓。

---

### Section 4: Findings — What the Framework Reveals

> 这一节是核心。三个 finding 由实验数据驱动，不预设结论。

#### Finding 1: Circuit Structure Bias（已确认）
- 现象：RL 方法在 L1/L2 上达到化学精度，但电路深度远超必要
- 根因：固定最大深度 + reward 仅由 energy error 驱动
- 展示方式：Pareto 图（energy error × CNOT count）

#### Finding 2: Fingerprint 与失败模式的相关性（数据驱动）
- 跑完实验后，将 Fingerprint 值与方法表现配对分析
- 展示方式：scatter plot（某 Fingerprint 值 vs 某方法指标）
- 🔴 具体规律待实验数据出来后再写，不预设
- 可增加一个补充子段：`full-space vs physical-sector fingerprints`
  - 目的不是替换主 benchmark 结果，而是检验 tier narrative 的化学稳健性
  - 特别关注：L3 的小 gap / degeneracy 解释、以及 G1-G4 在 sector projection 前后的变化

#### Finding 3: Policy 收敛不稳定（待测量）
- 展示方式：SR@chem vs episode 曲线（选 2-3 个有代表性的 case）
- 诊断问题：policy 是否真正收敛？何时收敛？是否会退化？
- 这个 finding 本身可能就是"RL for QAS 从未稳定收敛"

---

### Section 5: CRLQAS-STOP — Proof of Concept
- Finding 1（Circuit Structure Bias）是可修复的
- 加入 STOP 动作，agent 主动决定终止时机
- 对比实验：L1/L2 上 CRLQAS vs CRLQAS-STOP 的 Pareto 图

---

### Section 6: Discussion & Conclusion
- PSQASBench 对 RL for QAS 领域的启示
- Benchmark validity discussion：当前主任务采用 full-space Hamiltonian，但 physical-sector 分析表明，部分 fingerprint 叙事会受到 charge / spin sector 混合影响；后续 benchmark 可进一步探索 sector-aware 版本
- 🔴 开放问题列表（checkpoint 策略、policy 收敛性、L4+ 的根本挑战）
- 未来方向

---

## 叙事逻辑（两条故事线的衔接）

```
Section 3.1 分子集设计
    ↓ Fingerprint 描述了每个分子"应该"难在哪里
Section 4 实验结果
    ↓ 实验告诉我们方法"实际上"难在哪里
Section 4 Finding 2
    ↓ 对比两者：预期与实际是否吻合？不吻合处最有研究价值
Section 3.3 PCD
    ↓ 补充 Fingerprint 无法解释的部分（结构多样性）
Section 5 CRLQAS-STOP
    ↓ 框架不只是诊断工具，还能指导改进
```

---

## 悬而未决问题汇总（🔴）

1. **Checkpoint 统一方案**：last / 周期性 SR@chem 最高 / 仅报告训练过程指标，三选一
2. **PCD 计算时机**：依赖 checkpoint 方案，暂缓设计
3. **Fingerprint → 方法表现的具体相关性**：实验后数据驱动分析
4. **Hub Score / Asymmetry 的预测能力**：不预设，看数据
5. **physical-sector 分析放在正文还是附录**：取决于其对 tier narrative 的影响有多大
6. **L4+ 方法是否根本无法收敛**：待实验量化
7. **各方法的最大训练 episode 如何统一**：不同方法收敛速度差异大

---

## 方法覆盖范围

**RL 方法（主角）**：CRLQAS / PPO-QAS / Hybrid_REINFORCE / A2C-hybrid / RENEW / CRLQAS-STOP

**非 RL 参照（锚点）**：
- ADAPT-VQE（质量上界）
- Random Search（下界）
- QuantumDARTS（梯度类方法对照）
