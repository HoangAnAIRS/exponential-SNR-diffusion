# Exponential-SNR Diffusion

This repository contains the implementation of Exponential-SNR Diffusion. This model aims to enhance the signal-to-noise ratio (SNR) exponentially over time to improve the performance of diffusion models.

## Setup Instructions

To set up the environment for this project, follow these steps:

1. **Create a Conda Environment:**

    ```bash
    conda create -n diffusion python=3.8
    ```

2. **Activate the Conda Environment:**

    ```bash
    conda activate diffusion
    ```

## Repository Structure

- `improved_diffusion/`: Contains the source code for the Exponential-SNR Diffusion model.
- `datasets/`: Includes scripts to prepare datasets CIFAR-10.
- `notebooks/`: Jupyter notebooks for experimentation and visualization.
- `scripts/`: Shell and Python scripts for various utilities and tasks.

Besides, you can download trained checkpoints and sampled images from [this address](https://drive.google.com/drive/folders/1fXxR1AKijEeev_iIHDLWDcZw9h-cqyPE?usp=sharing)
## Getting Started

1. **Clone the Repository:**

    ```bash
    git clone https://github.com/HoangAnAIRS/exponential-SNR-diffusion.git
    cd exponential-SNR-diffusion
    ```

2. **Install Dependencies:**

    Ensure you have activated the `diffusion` environment:

    ```bash
    conda activate diffusion
    ```

    Then, install the required packages:
    ```bash
    pip install -e . 
    ```
    
    ```bash
    pip install -r requirements.txt
    ```

3. **Training the Model:**

    To train the model, use the following script:

    ```bash
    CUDA_VISIBLE_DEVICES=1
    python3 image_train.py \
    --data_dir /home/admin/workspace/user/improved-diffusion/datasets/cifar_train \
    --image_size 32 \
    --num_channels 128 \
    --num_res_blocks 3 \
    --learn_sigma False \
    --dropout 0.3 \
    --diffusion_steps 1000 \
    --noise_schedule linear \
    --lr 1e-4 \
    --batch_size 128 \
    --schedule_sampler "early" \
    --log_dir /home/admin/workspace/user/improved-diffusion/logs_early_snr_clamp_5.0
    ```
    or get inside scripts folder, change configs in sample.sh file and run:

    ```bash
    bash train.sh
    ```


4. **Sampling from the Model:**

    To generate samples from the trained model, use the provided script:

    ```bash
    python3 image_sample.py \
    --image_size 32 \
    --num_channels 128 \
    --num_res_blocks 3 \
    --dropout 0.3 \
    --diffusion_steps 1000 \
    --noise_schedule linear \
    --timestep_respacing ddim250 \
    --use_ddim True \
    --device cuda:2 \
    --model_path /home/admin/workspace/user/improved-diffusion/logs_early_snr_x_start/ema_0.9999_080000.pt \
    --output_npz_dir /home/admin/workspace/user/improved-diffusion/sampled_images_early_snr_x_start_ddim250
    ```

    or get inside scripts folder, change configs in sample.sh file and run:

    ```bash
    bash sample.sh
    ```

5. **Batch Sampling with Varying Models:**

    To perform batch sampling using different models, use the following script:

    ```bash
    ./batch_sample_varies_model.sh
    ```

    This script loops through each model and generates samples, storing the results in the specified output directory.

6. **Batch Sampling with Varying DDIM:**

    To perform batch sampling with varying DDIM timesteps, use the following script:

    ```bash
    ./batch_sample_varies_ddim.sh
    ```

    This script loops through different timestep respacing values and generates samples for each, storing the results in the respective output directories.

## Contact

For any questions or inquiries, please contact [Hoang An AIRS](mailto:hoangan@example.com).

