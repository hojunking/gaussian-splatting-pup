#!/bin/bash

# =================================================================================
# 0. 직접 지정할 scene 경로 목록
# =================================================================================
SCENES=(
    "/workdir/dataset/scannet/gt_pc/scene0000_02_qp27"
    "/workdir/dataset/scannet/gt_pc/scene0002_01_qp27"
    "/workdir/dataset/scannet/gt_pc/scene0011_00_qp27"
    "/workdir/dataset/scannet/gt_pc/scene0233_00_qp27"
    "/workdir/dataset/scannet/gt_pc/scene0499_00_qp27"
)
# =================================================================================
# 1. 공통 설정
# =================================================================================
GPU_ID=0
RESOLUTION=-1
SH_DEGREE=3
ITERATIONS=30000
TEST_ITERS="7000 30000"
SAVE_ITERS=30000
SAVE_PATH="experiments/gt_pc"
# =================================================================================
# 2. 지정한 SCENE 경로 반복 실행
# =================================================================================
for SCENE_PATH in "${SCENES[@]}"; do
    EXP_NAME=$(basename "${SCENE_PATH}")  # 폴더명 추출

    echo "==============================================="
    echo "[Processing Scene] ${EXP_NAME}"
    echo "==============================================="

    # -----------------------------------------------
    # 1) Training
    # -----------------------------------------------
    CUDA_VISIBLE_DEVICES=${GPU_ID} python train.py \
        -s "${SCENE_PATH}" \
        -m "${SAVE_PATH}/${EXP_NAME}" \
        --eval \
        --images "images" \
        --resolution ${RESOLUTION} \
        --data_device "cuda:${GPU_ID}" \
        --sh_degree ${SH_DEGREE} \
        --iterations ${ITERATIONS} \
        --test_iterations ${TEST_ITERS} \
        --save_iterations ${SAVE_ITERS} \
        --position_lr_init 0.00016 \
        --position_lr_final 0.0000016 \
        --position_lr_delay_mult 0.01 \
        --position_lr_max_steps 10000 \
        --feature_lr 0.0025 \
        --opacity_lr 0.05 \
        --scaling_lr 0.005 \
        --rotation_lr 0.001 \
        --percent_dense 0.01 \
        --lambda_dssim 0.2 \
        --densification_interval 100 \
        --opacity_reset_interval 3000 \
        --densify_from_iter 500 \
        --densify_until_iter 15000 \
        --densify_grad_threshold 0.0002 \
        --port 6010
    echo "Training finished for ${EXP_NAME}."

    # -----------------------------------------------
    # 2) Rendering
    # -----------------------------------------------
    python render.py \
        --source_path "${SCENE_PATH}" --skip_train \
        -m "${SAVE_PATH}/${EXP_NAME}"

    # # -----------------------------------------------
    # # 3) Metrics
    # # -----------------------------------------------
    python metrics.py --model_paths "${SAVE_PATH}/${EXP_NAME}" --qp

    python metric_tset_images.py -p "${SAVE_PATH}/${EXP_NAME}"

    echo "Completed processing for ${EXP_NAME}"
    echo ""
done

echo "==============================================="
echo "All listed scenes processed!"
echo "==============================================="
