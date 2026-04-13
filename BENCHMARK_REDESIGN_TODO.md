# PSQASBench Benchmark Redesign TODO

> 目的：整理合作者建议、当前数据集的 active-space 使用情况，以及基于这些建议对 6-level 分子体系的修改方向。
> 本文件是工作备忘录，不是最终论文表述。

---

## 1. Immediate TODO

1. 把 Level 4 的核心 fingerprint 从 `high-order ratio / k-local` 改成 `exact ground-state entanglement`。
2. 先实现单比特 Von Neumann entropy，并在 L4 候选分子上跑一遍。
3. 决定主 benchmark 是否继续保持 `full Hamiltonian` 叙事。
4. 如果主 benchmark 继续用 full Hamiltonian，则统一写清楚：`without charge or spin restriction`。
5. 如果要补 chemistry sanity-check，则把 `physical-sector` 分析作为补充诊断，而不是直接替换主 benchmark。
6. 把当前 6-level 缩减成“每层一个 anchor molecule”为优先目标。
7. 重新选择 L1 / L2 / L3 / L4 的单分子候选。
8. 把当前数据集的 `active_electrons / active_orbitals` 审计结果写入论文或附录。
9. 不把 `mapping / qubit tapering` 直接揉进 6-level 主轴，而作为附加变体轴处理。
10. L6 先固定为 `BeH2 8q / 10q / 12q` 主梯度，`14q` 作为可选扩展，不先写进主 benchmark 承诺。

---

## 2. Coauthor Suggestions

### 2.1 Metrics

1. 目前第一类 fingerprint 的方向是对的，而且和 diagonal dominance / Gershgorin circles 对应得很好。
2. 这部分的数学定义需要按合作者的新推导稍作修改，因此代码和分析都要同步更新。
3. 第一类指标除了刻画“接近对角形式”，还可以用来辅助判断 single-reference 与 multi-reference character。
4. 这在 quantum chemistry 语境下是重要解释维度，即使最后不是主指标，也值得保留。

### 2.2 k-local / High-order Ratio

1. 合作者对把 `>=4-body ratio` 作为 Level 4 核心依据持保留态度。
2. 原因是：从找基态的复杂性角度，`2-local` 及以上基本都已是 QMA-complete，单纯看 k-local 很难给出强有力的 ground-state difficulty justification。
3. k-local 更像 Hamiltonian simulation 的复杂度指标，不太适合作为 ground-state entanglement burden 的主解释。
4. 因此，Level 4 最好最终改成一个更直接反映 ground-state entanglement 的指标。

### 2.3 Entanglement Metric

1. 合作者认为目前找不到一个只靠 Pauli coefficients 就能稳妥反映 ground-state entanglement 的通用指标。
2. 因为本 benchmark 目前只做小系统，所以最直接可行的方法是用 exact ground state 本身来算纠缠。
3. 对每个 qubit 做 partial trace，得到 reduced density matrix。
4. 用 Von Neumann entropy 作为单比特纠缠度量。
5. 如有需要，可以进一步对 qubit pair 或更大 subsystem 重复这一分析。
6. 这与 DMRG 社区常见的 entanglement 视角一致，也方便未来扩展到 approximate ground state / 1-RDM proxy。
7. Renyi entropy 也可以考虑。
8. Mutual information 也可以考虑，但会混入 classical correlations。
9. 对当前论文来说，最简单稳妥的实现是先用单比特 Von Neumann entropy。

### 2.4 Charge / Spin Sector

1. 合作者认为当前关于 degeneracy 和 energy gap 的 chemistry 解释存在技术问题，因为目前分析的是 full Hamiltonian，没有限制正确的 charge / spin sector。
2. 当前看到的一些 degeneracy 很可能来自错误 charge state 或错误 spin sector 的态，而不是目标分子的物理简并。
3. 如果要做物理正确分析，应先投影到正确电子数子空间。
4. 最简单的电子数限制方式是只保留计算基中和 `active_electrons` 一致的比特串。
5. 进一步还可以按 alpha / beta electron number 限制 spin sector。
6. 然后在这个更小的 sector Hamiltonian 上重新做对角化，并重新计算 Energy gap / degeneracy。
7. 如果暂时不做完整修正，文中必须明确说明：当前分析是 `full Hamiltonian without charge or spin restriction`。

### 2.5 Degeneracy as a Level

1. 合作者并没有否定 degeneracy level 本身。
2. 相反，他认为 degeneracy 仍然是一个值得测试算法的 level。
3. 需要修正的是 degeneracy 的定义和分析方式，而不是把这一层删除。

### 2.6 Other Levels

1. 合作者认为除 L3 / L4 的问题外，其它 levels 整体设计是好的。
2. 也就是说，L1、L2、L5、L6 的方向可以继续保留。

### 2.7 Active Space / Frozen Core / Basis / Mapping

1. 合作者注意到当前数据集中有不少“分子看起来大，但 qubit 很少”的情况。
2. 这通常意味着使用了 active space truncation、frozen core、或其他轨道压缩手段。
3. 不同的轨道选择方法可能显著改变 Hamiltonian，因此 benchmark 表现有时可能是 Hamiltonian preprocessing 的问题，而不全是 QAS 的问题。
4. 合作者建议在数据库层面记录这些设定，必要时加入不同 active-space / frozen-core 版本。
5. basis set 也是一个重要变量，当前若大量使用 STO-3G，则应在论文中说明其局限性。
6. 更大的 basis（如 6-31G、6-311G 等）会显著增加 qubit 数和 Hamiltonian 结构复杂度。
7. mapping 和 qubit tapering 也值得考虑，但更适合作为附加变体轴而非直接改造 6-level 主轴。

### 2.8 Scalability

1. 合作者理解当前实验上限大约在 10 qubits 左右。
2. 但他提醒：更完整 basis / 更少截断时，很容易上到 30 qubits 量级。
3. 因此论文里最好回应这一限制，哪怕目前 benchmark 主体不能覆盖到这么大。

---

## 3. Dataset Audit: How The Current Molecules Were Truncated

### 3.1 What the code actually does

当前分子生成代码在 [mol_gen/prepare_molecules.py](mol_gen/prepare_molecules.py) 中统一使用：

- `active_electrons`
- `active_orbitals`

传给 `qchem.molecular_hamiltonian(...)`。

因此，从仓库代码本身可以确定的是：

1. 当前数据集统一使用的是 `active-space truncation`。
2. 代码里没有显式传入 `frozen_core=True/False`。
3. 代码里也没有显式指定“保留哪几个空间轨道、冻结哪几个 core orbitals”。
4. 因而我们可以记录 active-space 的规模，但不能仅凭当前代码精确还原“具体冻结了哪些轨道”。

### 3.2 Molecule-by-molecule audit

| Molecule | Symbols | Charge | Active Electrons | Active Orbitals | Qubits | Notes |
|------|------|------:|------:|------:|------:|------|
| L1_H2_Equil | H2 | 0 | 2 | 2 | 4 | 基本没有额外截断，最小 active space |
| L1_BH | BH | 0 | 2 | 3 | 6 | 对 BH 来说是明显的 active-space truncation |
| L2_BeH_Plus | BeH+ | +1 | 2 | 2 | 4 | 中等截断 |
| L2_LiH_Equil | LiH | 0 | 2 | 3 | 6 | 明显 active-space truncation |
| L2_BF | BF | 0 | 6 | 4 | 8 | 对 BF 来说是很激进的截断 |
| L3_HeH_Plus | HeH+ | +1 | 2 | 2 | 4 | 基本没有额外截断 |
| L3_CH2_Singlet | CH2 | 0 | 2 | 3 | 6 | 非常激进的截断 |
| L3_LiH_Stretch | LiH | 0 | 2 | 3 | 6 | 与 LiH equilibrium 相同 active space |
| L3_H3_Triangle | H3+ | +1 | 2 | 3 | 6 | 2e/3o active space |
| L4_H2_Stretch | H2 | 0 | 2 | 2 | 4 | 基本没有额外截断 |
| L4_H3_Linear | H3+ | +1 | 2 | 3 | 6 | 2e/3o active space |
| L4_H2O_StrongCorr | H2O | 0 | 4 | 4 | 8 | 对 H2O 来说是很激进的截断 |
| L5_H3_Linear | H3+ | +1 | 2 | 3 | 6 | 与 L4_H3_Linear 相同 active space |
| L5_H4_Chain | H4 | 0 | 4 | 4 | 8 | 基本不截断电子数 |
| L6_BeH2_Scalability | BeH2 | 0 | 2 | 5 | 10 | 明显 active-space truncation |

### 3.3 High-risk cases for chemistry interpretation

以下分子“化学上不小，但 qubit 数很小”，因此 benchmark 结果更容易受到 active-space 选择影响：

- BH
- BF
- CH2
- H2O
- BeH2

因此，如果某些 case 上方法表现异常，不能默认是 QAS 算法的问题，也可能是当前 active-space Hamiltonian 本身的结构较特殊。

---

## 4. How To Integrate Mapping / Qubit Tapering

当前不建议把 `mapping / tapering` 直接揉进 6-level 主轴。

更稳妥的做法是：

1. 主 benchmark 继续固定一个 canonical setting，例如 `Jordan-Wigner`。
2. 把 `Parity mapping`、`qubit tapering`、`sector-aware reductions` 作为附加变体轴。
3. 先在少量代表性 case 上做对照，而不是要求每个 level 都同时覆盖不同 mapping。

推荐的融入方式：

- 选择 1 个 L2 case 做 `JW vs parity/tapering` 对照；
- 选择 1 个 L3 case 做 `full-space vs restricted sector / tapered` 对照；
- 选择 1 个 L6 case 做 `scalability under different mappings` 对照。

这样可以回应合作者建议，但不会把 6-level 主叙事打散。

---

## 5. Proposed 6-Level Redesign

### 5.1 Guiding principles

1. 每层优先保留一个 anchor molecule。
2. 这个分子需要尽量代表该层核心 difficulty，而不是多个 difficulty 的混合。
3. 选择应同时考虑：
   - fingerprint 是否符合理论叙事；
   - chemistry 解释是否站得住；
   - 不同方法的实验表现是否有区分度。

### 5.2 Level-by-level judgement

#### L1 Minimalism

目标：

- 在不限制 depth 时，多数方法应较容易达到 chemical accuracy；
- 一旦约束 depth，应出现明显差异；
- Hamiltonian 应尽量接近对角、纠缠需求低。

当前判断：

1. `H2 (Equil.)` 太简单，不适合作为唯一 L1。
2. `BH` 概念上最像 L1，但如果多个 method 连不限制 depth 都达不到 chemical accuracy，则不适合作为唯一 L1 anchor。
3. `BeH2` 的小 basis 版本可以作为备选，但不能直接拿 10q BeH2 替换 L1，因为会把 scalability 混入 Minimalism。
4. 如果要测试 `BeH2` 作为 L1 备选，优先看 `BeH2_STO3G_6q`，但它的 asymmetry 比 BH 更明显，因此只是备选，不是最纯粹的 L1。

暂定建议：

- 先保留 `BH` 作为 L1 候选；
- 如果后续实验仍显示它在 unrestricted depth 下也无法稳定达到化学精度，则重新测试 `BeH2_STO3G_6q` 是否更适合作为实证上的 L1。

#### L2 Asymmetry / Interaction Hubs

目标：

- 需要一个 qubit importance 明显不均匀、但又不过度混入强关联和严重 degeneracy 的分子。

当前判断：

1. 如果只保留一个，我倾向于 `LiH (Equil.)`。
2. 原因是它是 neutral、6q、叙事干净，并且 Asymmetry / Hub Score 已经明显高于 L1。
3. `BF` 的 asymmetry 更强，但 mixed / high-order 也更强，污染更多。
4. `BeH+` 不建议作为唯一 L2，因为 charged system 会把 charge / sector 争议带进来。

暂定建议：

- L2 优先保留 `LiH (Equil.)`。

#### L3 Degeneracy / Stability

目标：

- 需要一个以 degeneracy / near-degeneracy 为核心 difficulty 的 case。

当前判断：

1. 如果 L3 只保留一个分子，我倾向于 `CH2`。
2. 在当前 full-space fingerprint 下，`CH2` 是最干净的 degeneracy case：
   - Gap 接近 0；
   - Z-only 很高；
   - high-order 很低；
   - 叙事上比 `LiH Stretch` 更少混入强关联与极端 asymmetry。
3. 如果目前不重做完整 charge / spin sector 修正，那么需要在正文里明确写成：
   - `full Hamiltonian without charge or spin restriction`

暂定建议：

- L3 保留 `CH2`。

#### L4 Representation / Correlation

目标：

- 测试 ansatz 的表达能力，核心难点应来自 ground-state entanglement / representation burden。

当前判断：

1. L4 不应继续以 `high-order ratio / k-locality` 作为主 justification。
2. 应先实现 exact ground-state entanglement fingerprint，再在候选中重新选择。
3. 第一轮建议在以下候选上跑 entropy 指标：
   - `H2 Stretch`
   - `H3 Linear`
   - `H2O`
4. `H2 Stretch` 太小，更像 sanity check。
5. 真正的单分子 anchor 很可能在 `H3 Linear` 和 `H2O` 之间产生。

暂定建议：

- L4 先不最终定分子；
- 先实现 Von Neumann entropy fingerprint，再在 `H3 Linear / H2O` 中二选一。

#### L5 Topology

当前判断：

1. 这一层建议保留。
2. 如果以后也想压缩成一个单分子，`H4 Chain` 可能比 `H3 Linear` 更能代表 routing pressure。

暂定建议：

- L5 保留现有方向；
- 是否只保留 `H4 Chain`，后续可再定。

#### L6 Scalability

当前判断：

1. L6 继续以 `BeH2` 为主没有问题。
2. 正式 benchmark 主梯度建议先写成：
   - `BeH2 8q / 10q / 12q`
3. `14q` 作为可选扩展或 stretch goal，不建议一开始就写成主 benchmark 必需项。
4. 原因是：
   - 14q 数据生成仍不稳定；
   - 训练显存 / 内存也已接近约束上限；
   - 现在就把 14q 写死，会增加 benchmark 不确定性。

暂定建议：

- L6 主体：`BeH2 8q / 10q / 12q`
- L6 optional extension：`BeH2 14q`

---

## 6. Main Narrative Options For Sector Analysis

### Option A: Keep full-space benchmark as main task

优点：

- 最接近当前 RLQAS / CRLQAS 实际优化的目标；
- 不需要重写整个 benchmark 主任务。

缺点：

- chemistry reader 会质疑 gap / degeneracy 的物理意义；
- 需要在文稿中非常清楚地写明：`without charge or spin restriction`。

适合当前阶段：

- 工作量最小；
- 能最快推进 benchmark 主结果。

### Option B: Keep full-space main task, add physical-sector diagnosis

优点：

- 兼顾 benchmark realism 与 chemistry validity；
- 与当前 `prepare_molecules_physical.py` 的方向一致；
- 最符合目前项目已有代码结构。

缺点：

- 需要额外分析与文稿解释。

适合当前阶段：

- 我认为这是最平衡的方案。

### Option C: Fully switch benchmark task to restricted sector

优点：

- 化学解释最干净。

缺点：

- 会直接改写 benchmark 任务定义；
- 与当前方法实际面对的目标不完全一致；
- 工作量最大。

适合当前阶段：

- 目前不建议。

---

## 7. Current Recommended Draft Of The 6 Levels

这是当前阶段最稳妥的一版工作性方案：

| Level | Working Choice | Status |
|------|------|------|
| L1 Minimalism | BH, backup candidate = BeH2_STO3G_6q | need experiment validation |
| L2 Asymmetry | LiH (Equil.) | relatively stable |
| L3 Degeneracy / Stability | CH2 | strongest current candidate |
| L4 Representation / Correlation | TBD after entropy analysis | not fixed yet |
| L5 Topology | keep current topology tier | can later compress |
| L6 Scalability | BeH2 8q / 10q / 12q, 14q optional | stable direction |

---

## 8. Next Actions

1. 实现 single-qubit Von Neumann entropy fingerprint。
2. 在 `H2 Stretch / H3 Linear / H2O` 上跑 entropy，决定 L4 anchor。
3. 对 `BH` 再做一次 L1 验证：
   - unrestricted depth 下是否能较稳定达化学精度；
   - depth-restricted 下是否确实拉开方法差距。
4. 如果 `BH` 不合适，则测试 `BeH2_STO3G_6q` 是否可作为 L1 备选。
5. 先用 `LiH (Equil.)` 锁定 L2。
6. 先用 `CH2` 锁定 L3。
7. 主 benchmark 暂定继续用 full-space Hamiltonian，但正文加上明确表述；
8. physical-sector 分析作为补充诊断继续保留。

