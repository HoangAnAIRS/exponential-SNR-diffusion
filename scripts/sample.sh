# CUDA_VISIBLE_DEVICES=3
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
--output_npz_dir /home/admin/workspace/user/improved-diffusion/sampled_images_early_snr_x_start_ddim250 \