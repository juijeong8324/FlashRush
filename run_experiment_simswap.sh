#!/bin/bash

ATTACK_TYPE=${1:-lab}      # lab | fgsm | pgd | fgsm_lab | pgd_lab
NUM_IMAGES=${2:-50}        # number of identity images to attack
NUM_TARGETS=${3:-5}        # number of fixed target faces

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="./logs/simswap_${ATTACK_TYPE}_${NUM_IMAGES}"
mkdir -p "$LOG_DIR"

echo "실험: main_simswap.py | attack_type: ${ATTACK_TYPE} | num_images: ${NUM_IMAGES} | num_targets: ${NUM_TARGETS} | 로그: ${LOG_DIR}"

nvidia-smi \
  --query-gpu=timestamp,index,name,utilization.gpu,utilization.memory,memory.used,memory.total \
  --format=csv -l 1 \
  > "${LOG_DIR}/gpu.csv" &
GPU_PID=$!

pidstat -u 1 -C python3 > "${LOG_DIR}/cpu.txt" &
CPU_PID=$!

python3 main_simswap.py \
  --arc_path SimSwap/arcface_model/arcface_checkpoint.tar \
  --G_path   SimSwap/checkpoints/people/latest_net_G.pth \
  --celeba_image_dir ./data/celeba/images \
  --attr_path        ./data/celeba/list_attr_celeba.txt \
  --result_dir       "./results/simswap_${ATTACK_TYPE}_${NUM_IMAGES}" \
  --attack_type      ${ATTACK_TYPE} \
  --num_id_images    ${NUM_IMAGES} \
  --num_target_images ${NUM_TARGETS} \
  > "${LOG_DIR}/result.txt" 2>&1

kill $GPU_PID $CPU_PID 2>/dev/null

echo ""
echo "✓ 로그 저장: ${LOG_DIR}/"
ls -lh "${LOG_DIR}/"
