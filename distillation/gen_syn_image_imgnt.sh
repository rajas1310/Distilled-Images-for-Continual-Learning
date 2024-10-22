CUDA_VISIBLE_DEVICES=0 python gen_syn_image.py \
    --dataset cifar10 \
    --diffusion_checkpoints_path stable-diffusion-v1-5/stable-diffusion-v1-5 \
    --guidance_scale 8 \
    --strength 0.7 \
    --ipc 20 \
    --km_expand 1 \
    --prototype_path ./prototypes/cifar10-ipc10-kmexpand1.json \
    --save_init_image_path ../data/distilled_data/ \
    --task 0