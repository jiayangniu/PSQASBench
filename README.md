# PSQASBench

Pauli String Quantum Architecture Search Benchmark — NeurIPS 2026 submission.

---

## 功能概览

PSQASBench 目前集成了两类 QAS 方法，并统一到同一套 benchmark 入口：

- `CRLQAS`：基于 DQN 的离散结构搜索
- `HyRLQAS`：基于 Hybrid REINFORCE / RENEW 的离散结构 + 连续参数联合搜索

框架提供了以下通用能力：

- 统一的 `main.py` 运行入口，按 `method / mol / seed` 组织实验
- 周期性评估指标，包括 `SR@chem`、`CNOT@chem`、`best/mean error`、`D_struct`、`D_func`
- 单环境与多环境并行训练
- 可切换的外部角度优化器
- Qulacs CPU / GPU 状态向量支持
- 结果与最优线路状态自动保存到 `results/`

---

## 环境配置

### 基础环境

```bash
conda env create -f environment.yml   # 或手动安装依赖
conda activate crlqas_env
```

---

## qulacs GPU 支持（QuantumStateGpu）

PyPI 版 `qulacs 0.6.13` 是 CPU-only build，不含 `QuantumStateGpu`。
若机器有 NVIDIA GPU + CUDA，需从源码编译才能启用 GPU state vector 加速。

> **已测试环境**：NVIDIA L4，CUDA 12.8，Python 3.10，conda `crlqas_env`

### 一次性编译步骤

**1. 安装编译依赖**

```bash
conda install -n crlqas_env -c conda-forge cuda-toolkit=12.8 boost boost-cpp -y
conda run -n crlqas_env pip install pybind11
```

**2. 克隆 qulacs 源码**

```bash
git clone --depth=1 https://github.com/qulacs/qulacs.git /tmp/qulacs
```

**3. 预下载 pybind11（FetchContent 在受限网络下无法自动拉取）**

```bash
git clone --depth=1 --branch v2.13.5 \
    https://github.com/pybind/pybind11.git /tmp/pybind11_src
```

**4. 打两处补丁（qulacs 源码 bug）**

```bash
# 补丁A：gpusim 改用动态 curand/cublas（conda 环境没有 _static.a）
sed -i 's/target_link_libraries(gpusim_static CUDA::cudart_static CUDA::curand_static CUDA::cublas_static)/target_link_libraries(gpusim_static CUDA::cudart_static CUDA::curand CUDA::cublas)/' \
    /tmp/qulacs/src/gpusim/CMakeLists.txt

# 补丁B：add_subdirectory out-of-tree 缺 binary dir 参数
sed -i 's|add_subdirectory(${pybind11_fetch_SOURCE_DIR})|add_subdirectory(${pybind11_fetch_SOURCE_DIR} ${CMAKE_BINARY_DIR}/_deps/pybind11_fetch-build)|' \
    /tmp/qulacs/CMakeLists.txt
```

**5. cmake configure**

```bash
mkdir -p /tmp/qulacs_build/_deps
cp -r /tmp/pybind11_src /tmp/qulacs_build/_deps/pybind11_fetch-src

conda run -n crlqas_env bash -c "
cd /tmp/qulacs_build
export CUDACXX=\$(which nvcc)
BOOST_DIR=\$(python -c 'import sysconfig; print(sysconfig.get_path(\"data\"))')
cmake /tmp/qulacs \
  -DPYTHON_EXECUTABLE=\$(which python) \
  -DPYTHON_SETUP_FLAG=Yes \
  -DUSE_GPU=Yes \
  -DCMAKE_C_COMPILER=gcc \
  -DCMAKE_CXX_COMPILER=g++ \
  -DCMAKE_BUILD_TYPE=Release \
  -DBOOST_ROOT=\$BOOST_DIR \
  -DFETCHCONTENT_FULLY_DISCONNECTED=ON
"
```

**6. 编译 + 安装**

```bash
conda run -n crlqas_env bash -c "
cd /tmp/qulacs_build
make -j\$(nproc) qulacs_core

SITE_PKG=\$(python -c 'import sysconfig; print(sysconfig.get_path(\"purelib\"))')
cp /tmp/qulacs_build/python/qulacs_core.cpython-310-x86_64-linux-gnu.so \$SITE_PKG/
cp -r /tmp/qulacs/pysrc/qulacs \$SITE_PKG/
"
```

**7. 验证**

```bash
conda run -n crlqas_env python -c "
from qulacs import QuantumStateGpu
s = QuantumStateGpu(4)
print('GPU qulacs OK')
"
```

安装成功后，`RLQAS/environment.py` 顶部的 try/except 会自动选择 `QuantumStateGpu`，无需修改代码。

### 补充说明

- 若 Python 版本不是 3.10，第6步的 `.so` 文件名中 `cpython-310` 需对应修改
- 编译约需 5–10 分钟（取决于 CPU 核心数）
- CPU-only 环境直接跳过本节，代码自动 fallback 到 `QuantumState`（CPU）

---

## 运行方式

```bash
cd PSQASBench
conda activate crlqas_env

# CRLQAS，L1 H2
python main.py --method crlqas --mol L1_H2_Equil_4q --seed 11111

# 使用 GPU（DQN 网络 + GPU state vector）
python main.py --method crlqas --mol L1_H2_Equil_4q --seed 11111 --device cuda:0
```

---

## 配置说明

每个方法的实验配置位于 `configs/`：

- `configs/crlqas/*.cfg`
- `configs/hyrlqas/*.cfg`

其中常用字段包括：

- `[general]`：训练轮数、评估频率、并行环境数、保存频率
- `[env]`：量子线路深度、奖励函数、阈值、curriculum 设置
- `[agent]`：策略网络结构、batch size、学习率、探索参数
- `[non_local_opt]`：每步外部角度优化器及其超参数

`num_parallel_envs > 1` 时会启用并行训练。对于 noiseless 场景，框架会按“相同线路结构”分组，复用 batched VQE kernel 进行并行能量评估。

---

## 外部优化器

框架当前支持以下 `optim_alg`：

- `COBYLA`
- `Rotosolve`
- `SPSA`
- `AdamSPSA`

统一通过配置项控制：

```ini
[non_local_opt]
method = scipy_each_step
optim_alg = SPSA
global_iters = 1000
```

### `COBYLA`

- 已完整接入框架，可直接用于 benchmark
- 适合做强基线
- 支持多环境并行训练
- 不支持当前这套“优化器侧 batched 加速”，因为 SciPy 黑盒搜索过程不能像 `Rotosolve/SPSA` 一样整齐打包为固定批次

### `Rotosolve`

- 已支持单环境与并行环境
- 在并行训练下支持 grouped batched optimizer 加速
- 适合无噪声、旋转参数较多但单参数解析更新仍有效的场景

并行加速开关示例：

```ini
[non_local_opt]
method = scipy_each_step
optim_alg = Rotosolve
rotosolve_sweeps = 1
global_batched_rotosolve = 1
```

### `SPSA` / `AdamSPSA`

- 已支持单环境与并行环境
- 已支持 grouped batched optimizer 加速
- 更适合希望保留并行优化器加速、同时避免 `Rotosolve` 解析更新限制的场景

推荐配置示例：

```ini
[non_local_opt]
method = scipy_each_step
optim_alg = AdamSPSA
global_iters = 1000
global_batched_spsa = 1

a = 0.05
alpha = 0.602
c = 0.1
gamma = 0.101
lamda = 100
beta_1 = 0.9
beta_2 = 0.999
```

说明：

- `SPSA` 使用标准 SPSA 更新
- `AdamSPSA` 在 SPSA 梯度估计基础上加入 Adam 风格动量与二阶矩归一化
- `global_batched_spsa = 1` 时，并行 runner 会对相同线路结构的环境做分组，并用 batched VQE 一次评估 `theta + ck * delta` 与 `theta - ck * delta`

---

## 并行加速机制

框架中的“并行”分成两层：

1. 训练并行：通过 `num_parallel_envs` 同时推进多个环境
2. 优化器并行：对支持固定批次探测的优化器进行 grouped batched 加速

当前支持情况如下：

- `COBYLA`：支持训练并行，不支持优化器 batched 加速
- `Rotosolve`：支持训练并行，支持优化器 batched 加速
- `SPSA`：支持训练并行，支持优化器 batched 加速
- `AdamSPSA`：支持训练并行，支持优化器 batched 加速

因此，如果目标是“更换优化器但保留目前并行加速思路”，建议优先尝试 `SPSA` 或 `AdamSPSA`。
