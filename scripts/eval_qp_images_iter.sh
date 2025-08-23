#!/bin/bash

# =================================================================================
# 0. 직접 지정할 scene 경로 목록
# =================================================================================
SCENES=(
    "/workdir/dataset/scannet/scene0000_02_qp27"
    "/workdir/dataset/scannet/scene0002_01_qp27"
    "/workdir/dataset/scannet/scene0011_00_qp27"
    "/workdir/dataset/scannet/scene0233_00_qp27"
    "/workdir/dataset/scannet/scene0499_00_qp27"
)

# =================================================================================
# 2. 지정한 SCENE 경로 반복 실행
# =================================================================================
for SCENE_PATH in "${SCENES[@]}"; do
    EXP_NAME=$(basename "${SCENE_PATH}")  # 폴더명 추출

    echo "==============================================="
    echo "[Processing Scene] ${EXP_NAME}"
    echo "==============================================="

    # -----------------------------------------------
    # 3) Metrics
    # -----------------------------------------------
    python metric_images.py -p "dataset/scannet/${EXP_NAME}"

    echo "Completed processing for ${EXP_NAME}"
    echo ""
done

echo "==============================================="
echo "All listed scenes processed!"
echo "==============================================="
