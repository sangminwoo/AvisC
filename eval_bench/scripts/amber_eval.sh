#!/bin/bash


## set below
####################################################
seed=42
conv="llava_v1"
batch_size=1
model="llava" # llava | qwen-vl | instructblip
use_avisc=True
layer_gamma=0.5
masking_scheme="zeros"			# "10:1" or "-1"
lamb=1.0
gpus=1
max_token=64
log_path="path/to/log/"
cd_alpha=2.5
amber_path="path/to/AMBER"
use_cd=False
use_m3id=False
json_path="path/to/AMBER/data/query/query_all.json"
data_path="path/to/dataset/AMBER/image"
####################################################

result_json_path="${log_path}/.json"

export CUDA_VISIBLE_DEVICES=${gpus}
python ./eval_bench/amber_eval_${model}.py \
    --seed ${seed} \
    --model-path ${model_path} \
    --json_path ${json_path} \
    --data_path ${data_path} \
    --log_path ${log_path} \
    --conv-mode ${conv} \
    --batch-size ${batch_size} \
    --use_avisc ${use_avisc} \
    --layer_gamma ${layer_gamma} \
    --masking_scheme ${masking_scheme} \
    --lamb ${lamb} \
    --use_cd ${use_cd} \
    --exp_description ${exp_description} \
    --max_token ${max_token} \
    --cd_alpha ${cd_alpha} \
    --use_m3id ${use_m3id} \

python ./experiments/AMBER/inference.py \
    --inference_data ${result_json_path} \
    --word_association "${amber_path}/data/relation.json" \
    --safe_word "${amber_path}/data/safe_words.txt"\
    --annotation "${amber_path}/data/annotations.json"\
    --metrics "${amber_path}/data/metrics.txt"\
    --evaluation_type a \


