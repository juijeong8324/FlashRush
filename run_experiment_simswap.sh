#!/bin/bash

SCRIPT=${1:-main_simswap}          # main_simswap | main_simswap2 | main_simswap3
NUM_IMAGES=${2:-50}                # number of identity images to attack
ATTACK_ITERS=${3:-100}             # number of attack iterations
EPSILON=${4:-0.5}                  # perturbation epsilon
TARGET_IMAGE=${5:-SimSwap/crop_224/6.jpg}

# main_simswap = baseline (MPI 없음), main_simswap2/3 = MPI
if [ "$SCRIPT" = "main_simswap" ]; then
    RUN_PREFIX="taskset -c 0,1 python3"
else
    RUN_PREFIX="taskset -c 0,1 mpirun -np 2 python3"
fi

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="./logs/${SCRIPT}_${NUM_IMAGES}_${ATTACK_ITERS}"
mkdir -p "$LOG_DIR"

echo "실험: ${SCRIPT}.py | num_images: ${NUM_IMAGES} | attack_iters: ${ATTACK_ITERS} | epsilon: ${EPSILON} | target: ${TARGET_IMAGE} | 로그: ${LOG_DIR}"

nvidia-smi \
  --query-gpu=timestamp,index,name,utilization.gpu,utilization.memory,memory.used,memory.total \
  --format=csv -l 1 \
  > "${LOG_DIR}/gpu.csv" &
GPU_PID=$!

pidstat -u 1 -C python3 > "${LOG_DIR}/cpu.txt" &
CPU_PID=$!

$RUN_PREFIX ${SCRIPT}.py \
  --arc_path SimSwap/arcface_model/arcface_checkpoint.tar \
  --G_path   SimSwap/checkpoints/people/latest_net_G.pth \
  --celeba_image_dir ./data/celeba/images \
  --attr_path        ./data/celeba/list_attr_celeba.txt \
  --result_dir       "./results/${SCRIPT}_${NUM_IMAGES}_${ATTACK_ITERS}" \
  --num_id_images    ${NUM_IMAGES} \
  --attack_iters     ${ATTACK_ITERS} \
  --epsilon          ${EPSILON} \
  --target_image     ${TARGET_IMAGE} \
  > "${LOG_DIR}/result.txt" 2>&1

kill $GPU_PID $CPU_PID 2>/dev/null

echo ""
echo "✓ 로그 저장: ${LOG_DIR}/"
ls -lh "${LOG_DIR}/"
