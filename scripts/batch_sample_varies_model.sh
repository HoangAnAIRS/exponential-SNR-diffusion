#!/bin/bash

# Directory containing the models
MODEL_DIR="/home/admin/workspace/user/improved-diffusion/logs_uniform"

# Output directory
OUTPUT_DIR="/home/admin/workspace/user/improved-diffusion/sampled_images_uniform_noise_ddim100"

# Common parameters
IMAGE_SIZE=32
NUM_CHANNELS=128
NUM_RES_BLOCKS=3
DROPOUT=0.3
DIFFUSION_STEPS=1000
NOISE_SCHEDULE="linear"
TIMESTEP_RESPACING="ddim100"
DEVICE="cuda:3"
USE_DDIM="True"

# List of model names
MODEL_NAMES=("ema_0.9999_250000.pt")

# Loop through each model name
for MODEL_NAME in "${MODEL_NAMES[@]}"
do
  # Construct the full model path
  MODEL_PATH="${MODEL_DIR}/${MODEL_NAME}"

  # Run the python command
  python3 image_sample.py \
  --image_size ${IMAGE_SIZE} \
  --num_channels ${NUM_CHANNELS} \
  --num_res_blocks ${NUM_RES_BLOCKS} \
  --dropout ${DROPOUT} \
  --diffusion_steps ${DIFFUSION_STEPS} \
  --noise_schedule ${NOISE_SCHEDULE} \
  --timestep_respacing ${TIMESTEP_RESPACING} \
  --use_ddim ${USE_DDIM} \
  --device ${DEVICE} \
  --model_path ${MODEL_PATH} \
  --output_npz_dir ${OUTPUT_DIR}
done
