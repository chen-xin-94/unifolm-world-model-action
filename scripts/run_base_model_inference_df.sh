#!/bin/bash

model_name=base_model_df
ckpt=checkpoints/unifolm_wma_base.ckpt
config=configs/inference/base_model_inference_df.yaml
res_dir="output/base_model_df"
seed=123

CUDA_VISIBLE_DEVICES=0 python3 scripts/evaluation/base_model_inference.py \
--seed ${seed} \
--ckpt_path $ckpt \
--config $config \
--savedir "${res_dir}/videos" \
--bs 1 --height 320 --width 512 \
--unconditional_guidance_scale 1.0 \
--ddim_steps 16 \
--ddim_eta 1.0 \
--prompt_dir "examples/base_model_prompts_df" \
--text_input \
--video_length 16 \
--timestep_spacing 'uniform_trailing' \
--guidance_rescale 0.7 \
--perframe_ae
