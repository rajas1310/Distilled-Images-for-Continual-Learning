CUDA_VISIBLE_DEVICES=0 python gen_syn_image.py \
    --dataset mnist \
    --diffusion_checkpoints_path stable-diffusion-v1-5/stable-diffusion-v1-5 \
    --guidance_scale 8 \
    --strength 0.7 \
    --ipc 50 \
    --km_expand 1 \
    --prototype_path ./prototypes/mnist-ipc50-kmexpand1-task8.json \
    --save_init_image_path ../data/distilled_data/mnist \
    --task 8