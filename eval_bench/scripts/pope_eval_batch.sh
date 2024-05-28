#!/bin/bash


## set below
####################################################
seed=42
dataset_name="coco"  # coco | aokvqa | gqa
type="random" # random | popular | adversarial
model="llava" # llava | qwen-vl | instructblip
use_avisc=False
layer_gamma=0.5
masking_scheme="zeros"			# "10:1" or "-1"
lamb=1.0
gpus=3
max_token=64
log_path="path/to/log/pope/"
cd_alpha=1.0
cd_beta=0.1
cd_alpha_list=(3.0)

model_path="/path/to/the/checkpoints/"
pope_path="path/to/experiments/data/POPE/${dataset_name}/${dataset_name}_pope_${type}.json"
data_path="/path/to/the/data/coco/val2014"
####################################################



export CUDA_VISIBLE_DEVICES=${gpus}
python ./eval_bench/pope_eval_${model}b.py\
--seed ${seed} \
--model-path ${model_path} \
--pope_path ${pope_path} \
--data_path ${data_path} \
--log_path ${log_path} \
--conv-mode ${conv} \
--batch-size ${batch_size} \
--use_avisc ${use_avisc} \
--layer_gamma ${layer_gamma} \
--masking_scheme ${masking_method} \
--lamb ${lamb} \
--use_cd ${use_cd} \
--exp_description ${exp_description} \
--max_token ${max_token} \
--cd_alpha ${cd_alpha} \
--cd_beta ${cd_beta} \
