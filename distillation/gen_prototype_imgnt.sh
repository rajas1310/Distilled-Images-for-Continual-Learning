#!/bin/bash
conda init bash\
conda activate D4M \

CUDA_VISIBLE_DEVICES=0 python gen_prototype.py \
    --batch_size 10 \
    --data_dir ../../cifar10 \
    --dataset cifar10 \
    --diffusion_checkpoints_path stable-diffusion-v1-5/stable-diffusion-v1-5 \
    --ipc 10 \
    --km_expand 1 \
    --label_file_path ./label-prompt/CIFAR-10_labels.txt \
    --save_prototype_path ./prototypes