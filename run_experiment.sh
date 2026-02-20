#!/bin/bash

SCRIPT=${1:-main}   # main | main2 | main3

# main은 baseline (MPI 없음), main2/main3은 MPI
if [ "$SCRIPT" = "main" ]; then
    RUN_PREFIX="python3"
else
    RUN_PREFIX="mpirun -np 2 python3"
fi

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="./logs/${SCRIPT}_${TIMESTAMP}"
mkdir -p "$LOG_DIR"

echo "실험: ${SCRIPT}.py | 로그: ${LOG_DIR}"

nvidia-smi \
  --query-gpu=timestamp,index,name,utilization.gpu,utilization.memory,memory.used,memory.total \
  --format=csv -l 1 \
  > "${LOG_DIR}/gpu.csv" &
GPU_PID=$!

pidstat -u 1 -C python3 > "${LOG_DIR}/cpu.txt" &
CPU_PID=$!

$RUN_PREFIX ${SCRIPT}.py \
  --mode test \
  --image_size 256 \
  --c_dim 5 \
  --selected_attrs Black_Hair Blond_Hair Brown_Hair Male Young \
  --model_save_dir='stargan_celeba_256/models' \
  --result_dir="./results/${SCRIPT}" \
  --test_iters 200000 \
  --attack_iters 100 \
  --batch_size 1 \
  2>&1 | tee "${LOG_DIR}/result.txt"

kill $GPU_PID $CPU_PID 2>/dev/null

echo ""
echo "✓ 로그 저장: ${LOG_DIR}/"
ls -lh "${LOG_DIR}/"
