#!/bin/bash


## set below
####################################################
seed=42
model="llava" # llava | qwen-vl | instructblip
use_avisc=false
use_cd=False
gpus=0
max_token=64
cd_alpha=2.5
cd_beta=0.1
model_path="/path/to/the/checkpoints/llava-v1.5-7b"
pope_path="path/to/dataset/llava-bench-in-the-wild/questions.jsonl"
data_path="path/to/dataset/llava-bench-in-the-wild/images"
log_path="path/to//llava_bench/.json"
conv="llava_v1"
batch_size=1
####################################################

export CUDA_VISIBLE_DEVICES=${gpus}
python ./eval_bench/llava_bench_llava.py \
--seed ${seed} \
--model-path ${model_path} \
--question-file ${pope_path} \
--image-folder ${data_path} \
--answers-file ${log_path} \
--conv ${conv} \
--use_avisc ${use_avisc} \
--use_cd ${use_cd} \
--max_token ${max_token} \
--cd_alpha ${cd_alpha} \
--cd_beta ${cd_beta} \

