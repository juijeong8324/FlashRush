#!/bin/bash

SCRIPT=${1:-main}          # main | main2 | main3
ATTACK_TYPE=${2:-lab}      # lab | fgsm | pgd
NUM_IMAGES=${3:-50}        # number of images to process
ATTACK_ITERS=${4:-100}     # number of attack iterations

# main은 baseline (MPI 없음), main2/main3은 MPI
# 기존 뉴런환경과 비교하기 위함
if [ "$SCRIPT" = "main" ]; then
    RUN_PREFIX="taskset -c 0,1 python3"
else
    RUN_PREFIX="taskset -c 0,1 mpirun -np 2 python3"
fi

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="./logs/${SCRIPT}_${ATTACK_TYPE}_${ATTACK_ITERS}"
mkdir -p "$LOG_DIR"

echo "실험: ${SCRIPT}.py | attack_type: ${ATTACK_TYPE} | num_images: ${NUM_IMAGES} | attack_iters: ${ATTACK_ITERS} | 로그: ${LOG_DIR}"

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
  --result_dir="./results/${SCRIPT}_${ATTACK_TYPE}_${NUM_IMAGES}_${ATTACK_ITERS}" \
  --test_iters 200000 \
  --attack_type ${ATTACK_TYPE} \
  --num_images ${NUM_IMAGES} \
  --attack_iters ${ATTACK_ITERS} \
  --batch_size 1 \
  > "${LOG_DIR}/result.txt" 2>&1

kill $GPU_PID $CPU_PID 2>/dev/null

echo ""
echo "✓ 로그 저장: ${LOG_DIR}/"
ls -lh "${LOG_DIR}/"
