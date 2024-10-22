git clone https://github.com/rajas1310/diffusers_for_D4M.git \
git clone https://github.com/rajas1310/DL566.git \
pip install torch==2.4.1 torchvision==0.19.1
cd diffusers_for_D4M \
pip install -e '.[torch]' \
pip install transformers scikit-learn ipdb
cd ../DL566 \

CUDA_VISIBLE_DEVICES=0 python gen_prototype.py \
    --batch_size 10 \
    --data_dir ../../cifar10 \
    --dataset cifar10 \
    --diffusion_checkpoints_path stable-diffusion-v1-5/stable-diffusion-v1-5 \
    --ipc 10 \
    --km_expand 1 \
    --label_file_path ./label-prompt/CIFAR-10_labels.txt \
    --save_prototype_path ./prototypes