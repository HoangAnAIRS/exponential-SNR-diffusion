#!/bin/bash

# Directory containing the models
MODEL_DIR="/home/admin/workspace/user/improved-diffusion/logs_early_snr"

# Output directory
OUTPUT_DIR="/home/admin/workspace/user/improved-diffusion/sampled_images_early_snr_noise_ddim"

# Common parameters
IMAGE_SIZE=32
NUM_CHANNELS=128
NUM_RES_BLOCKS=3
DROPOUT=0.3
DIFFUSION_STEPS=1000
NOISE_SCHEDULE="linear"
DEVICE="cuda:1"
USE_DDIM="True"

# Fixed model name
MODEL_NAME="ema_0.9999_250000.pt"

# List of timestep respacing values
TIMESTEP_RESPACING_VALUES=("ddim10" "ddim50" "ddim100")

# Loop through each timestep respacing value
for TIMESTEP_RESPACING in "${TIMESTEP_RESPACING_VALUES[@]}"
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
  --output_npz_dir "${OUTPUT_DIR}_${TIMESTEP_RESPACING}"
done
