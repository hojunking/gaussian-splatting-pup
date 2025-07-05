#!/bin/bash

# =================================================================================
# 1. 경로 및 실험 이름 설정
# =================================================================================
# scene: scene_path
SCENE_PATH="/workdir/dataset/scannet/scannet_samples_pre/scene0000_00"

# model: model_dir 와 train: exp_name 을 조합
EXP_NAME="scene0029_02"


# 결과물을 저장할 디렉토리 생성
mkdir -p $OUTPUT_PATH

# =================================================================================
# 2. train.py 실행
# =================================================================================

CUDA_VISIBLE_DEVICES=0 python train.py \
    -s /scannet/scene0011_00 \
    -m experiments/ \
    --eval \
    --images "images" \
    --resolution -1 \
    --data_device "cuda:0" \
    --sh_degree 3 \
    --iterations 10000 \
    --test_iterations 1000 7000 \
    --save_iterations 10000 \
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
    --densification_interval 1000 \
    --opacity_reset_interval 2000 \
    --densify_from_iter 500 \
    --densify_until_iter 10000 \
    --densify_grad_threshold 0.0002

echo "Training finished for ${EXP_NAME}."