#!/usr/bin/env bash
# =============================================================================
# GPU Optimizer Benchmark: 8q BeH2 (L6_BeH2_631G)
# 5 configurations × 200 episodes, seed=11111
#
# 对比维度:
#   [1] Cobyla(cpu, k=1)  vs  Cobyla(gpu, k=1)   → GPU 单次 eval 加速
#   [2] Cobyla(gpu, k=1)  vs  Cobyla(gpu, k=10)  → K 扩展代价
#   [3] Cobyla(gpu, k=10) vs  Rotosolve(gpu,k=10) vs PSRAdam(gpu,k=10)
#                                                  → 批量 GPU 优化器优势
#
# 运行方式:
#   bash run_optimizer_bench.sh
#
# 结果路径:
#   results/crlqas/L6_BeH2_631G_8q/<config>/seed11111/run.log
# =============================================================================

set -e

PYTHON=/home/ubuntu/miniconda3/envs/crlqas_env/bin/python
SEED=11111
MOL=L6_BeH2_6311G_10q
DEVICE=cuda

CONFIGS=(
    "bench_10q_cobyla_cpu_k1"
    "bench_10q_cobyla_gpu_k1"
    "bench_10q_cobyla_gpu_k10"
    "bench_10q_rotosolve_gpu_k10"
    "bench_10q_psradam_gpu_k10"
)

LABELS=(
    "Cobyla(cpu,  k=1 )"
    "Cobyla(gpu,  k=1 )"
    "Cobyla(gpu,  k=10)"
    "Rotosolve(gpu,k=10)"
    "PSRAdam(gpu, k=10)"
)

echo "========================================================"
echo "  Optimizer Benchmark — 8q BeH2, 200 episodes, seed=${SEED}"
echo "========================================================"
echo ""

for i in "${!CONFIGS[@]}"; do
    CFG="${CONFIGS[$i]}"
    LABEL="${LABELS[$i]}"

    echo "--------------------------------------------------------"
    echo "  [$(( i+1 ))/5]  ${LABEL}"
    echo "  config: ${CFG}"
    echo "--------------------------------------------------------"

    START=$(date +%s)

    $PYTHON main.py \
        --method crlqas \
        --mol    "${MOL}" \
        --config "${CFG}.cfg" \
        --seed   "${SEED}" \
        --device "${DEVICE}" \
        2>&1 | tee "results/crlqas/${MOL}/${CFG}/seed${SEED}/run.log"

    END=$(date +%s)
    ELAPSED=$(( END - START ))

    echo ""
    echo "  ✓ 完成  总耗时=${ELAPSED}s  ($(( ELAPSED/60 ))m$(( ELAPSED%60 ))s)"
    echo ""
done

echo "========================================================"
echo "  所有实验完成，汇总 elapsed 时间："
echo "========================================================"
for i in "${!CONFIGS[@]}"; do
    CFG="${CONFIGS[$i]}"
    LABEL="${LABELS[$i]}"
    LOG="results/crlqas/${MOL}/${CFG}/seed${SEED}/run.log"
    if [ -f "$LOG" ]; then
        # 抓最后一行 ep 日志里的 elapsed
        LAST=$(grep "^\[ep " "$LOG" | tail -1)
        echo "  ${LABEL}: ${LAST}"
    fi
done
echo ""
echo "详细 log 路径: results/crlqas/${MOL}/<config>/seed${SEED}/run.log"
