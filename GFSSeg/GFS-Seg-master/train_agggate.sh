#!/bin/sh

export PYTHONPATH=./
dataset=$1
exp_name=$2
gpu_id=$3
exp_dir=exp/${dataset}/${exp_name}
model_dir=${exp_dir}/model
result_dir=${exp_dir}/result
config=config/${dataset}/${dataset}_${exp_name}.yaml

mkdir -p ${model_dir} ${result_dir}
now=$(date +"%Y%m%d_%H%M%S")
cp train_agggate.sh train_agggate.py ${config} ${exp_dir}

CUDA_VISIBLE_DEVICES=${gpu_id} python3 -u train_agggate.py --config=${config} 2>&1 | tee ${result_dir}/train-$now.log
