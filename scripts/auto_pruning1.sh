#!/bin/bash

# =================================================================================
# 스크립트 설정
# =================================================================================
# 1. 오류 발생 시 즉시 스크립트 종료
set -e

# 2. Scannet 원본 데이터가 있는 루트 경로
SCANNET_ROOT="/scannet"

# 3. 시작 인덱스 설정 (없으면 0부터 시작)
START_INDEX=${1:-0}

# =================================================================================
# 메인 실행 로직
# =================================================================================

# 1. /scannet 폴더에서 scene 목록을 찾아 정렬
SCENES=($(find "${SCANNET_ROOT}" -mindepth 1 -maxdepth 1 -type d -name "scene*" | sort))

# 2. 찾은 scene 목록을 순회하며 파이프라인 실행
TOTAL_SCENES=${#SCENES[@]}
echo "Total scenes found: ${TOTAL_SCENES}. Starting from index ${START_INDEX}."

for i in $(seq $START_INDEX $(($TOTAL_SCENES - 1))); do
    
    scene_full_path=${SCENES[$i]}
    scene_name=$(basename "$scene_full_path")

    echo "----------------------------------------------------------------------"
    echo "--- [$(($i + 1))/${TOTAL_SCENES}] Delegating to pruning1.sh for: ${scene_name}"
    echo "----------------------------------------------------------------------"

    # pruning1.sh 스크립트를 scene_name을 인자로 하여 호출
    ./scripts/pruning1.sh "${scene_name}"

done

echo "======================================================================"
echo "All processing complete."
echo "======================================================================"