
set -e

if [ -z "$1" ]; then
    echo "Error: No scene_name provided to pruning1.sh"
    exit 1
fi

# 첫 번째 인자($1)를 scene_name 변수에 할당합니다.
scene_name=$1
#scene_name='scene0011_01'
source_path='/scannet/'${scene_name}
orig_path='experiments/'${scene_name}
start_iteration=30000

# PRUNE TYPE AND PERCENTAGES
prune_type=fisher
round1=80
round2=50


# PRUNE ROUND 1
directory1=./experiments/${scene_name}/$round1
mkdir -p $directory1/point_cloud/iteration_${start_iteratxion}
# cp -r $orig_path/cameras.json $directory1/cameras.json
# cp -r $orig_path/cfg_args $directory1/cfg_args
CUDA_VISIBLE_DEVICES=0 python prune_finetune.py \
    -s $source_path \
    -m $directory1 \
    --start_pointcloud /point_cloud/${scene_name}/ckpts/point_cloud_30000.ply \
    --eval \
    --prune_percent 0.$round1 \
    --position_lr_max_steps 35000 \
    --iterations 35000 \
    --save_iterations 35000 \
    --checkpoint_iterations 0 \
    --test_iterations 0 \
    --prune_type $prune_type \
    --fisher_resolution 4 \
    --first_iter 30000 \
    --port 6071 
    #--start_pointcloud ${orig_path}/point_cloud/iteration_30000/point_cloud.ply \

# PRUNE ROUND 2
directory2=$directory1\_$round2
mkdir -p $directory2
# cp -r $orig_path/cameras.json $directory2/cameras.json
# cp -r $orig_path/cfg_args $directory2/cfg_args
CUDA_VISIBLE_DEVICES=0 python prune_finetune.py \
    -s $source_path \
    -m $directory2 \
    --start_pointcloud ${directory1}/point_cloud/iteration_35000/point_cloud.ply \
    --eval \
    --prune_percent 0.$round2 \
    --position_lr_max_steps 35000 \
    --iterations 35000 \
    --save_iterations 35000 \
    --checkpoint_iterations 0 \
    --test_iterations 0 \
    --prune_type $prune_type \
    --fisher_resolution 4 \
    --first_iter 30000 \
    --port 6071 \

# COLLECT METRICS
python render.py \
    --source_path $source_path --skip_train \
    -m $directory2
python metrics.py --model_paths $directory2
