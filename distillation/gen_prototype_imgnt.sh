CUDA_VISIBLE_DEVICES=0 python gen_prototype.py \
    --batch_size 10 \
    --data_dir ../../cifar10 \
    --dataset cifar10 \
    --diffusion_checkpoints_path stable-diffusion-v1-5/stable-diffusion-v1-5 \
    --ipc 50 \
    --km_expand 1 \
    --save_prototype_path ./prototypes \
    --task 0