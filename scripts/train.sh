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
--log_dir /home/admin/workspace/user/improved-diffusion/logs_early_snr_clamp_5.0 \