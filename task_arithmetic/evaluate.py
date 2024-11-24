import torch
import numpy as np

from apply_ta import get_model

# from vit_baseline import ViT_LoRA
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from torchvision import transforms
import torch
from logger_utils import Logger
import sys, os, time
from net import ResNet, test, fit
from data import *

import argparse

parser = argparse.ArgumentParser()


parser.add_argument('-e', '--epochs', type=int, default=50)
parser.add_argument('-bs', '--batch-size', type=int, default=16)
parser.add_argument('-lr', '--lr', type=float, default=5e-6)
parser.add_argument('-wd', '--weight-decay', type=float, default=1e-6)
parser.add_argument('-nw','--num-workers', type=int, default=2)
parser.add_argument('--test-interval', type=int, default=1)
parser.add_argument('--syn', action="store_true")
parser.add_argument('--device', type=str, default="cuda:0")
parser.add_argument('--model', type=str, default="resnet18")
# parser.add_argument('--seed', type=int, default=42)

parser.add_argument('-d', '--dataset', type=str, default='oxfordpet')
parser.add_argument('-ddir', '--data-dir', type=str, default='../data')
parser.add_argument('-odir', '--output-dir', type=str, default='./output')
parser.add_argument('-midir', '--model-input-dir', type=str, default='./data')
parser.add_argument('-t', '--tasknum', type=int)
parser.add_argument('-tot', '--total-tasks', type=int)
parser.add_argument('-scoef', '--scaling-coef', type=float, default=0.25)
parser.add_argument('--tag', type=str, default="")

parser.add_argument('-nc', '--num-classes', type=int, default=None)
args = parser.parse_args()

args.output_dir = f"{args.output_dir}/{args.model}_{args.dataset}"

if args.num_classes == None:
    if args.dataset == 'mnist' or args.dataset == 'cifar10':
        args.num_classes = 10

# if args.num_classes == None:
#     args.num_classes = 10 if args.data=='svhn' else( 37 if args.data=='oxfordpet' else 102)



if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir, exist_ok=True)
    
sys.stdout = Logger(os.path.join(args.output_dir, 'logs-evaluate-{}-TACL.txt'.format(args.dataset)))

print(args)

# img_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")



# get pretrained model
# pretrained_model = ResNet(args.num_classes, args.device, args.model)
# torch.save(pretrained_model, f"{args.output_dir}/resnet18-pretrained.pth")
pretrained_path = '/content/drive/MyDrive/DL 566 Project/colab_output/resnet18/resnet18-pretrained.pth'
pretrained_model = torch.load(pretrained_path)

test_all_tasks = list()

for task_idx in range(args.total_tasks):
    args.tasknum = task_idx
    testset = TaskwiseOriginalDataset(args, split='test').get_dataset()

    testloader = torch.utils.data.DataLoader(
            testset, batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers
        )
    print(f"Length of {task_idx}th test dataset", len(testset))
    test_all_tasks.append(testloader)

final_model = get_model(args, pretrained_path, list_of_task_checkpoints=[f"{args.model_input_dir}/{args.model}_task_{i}_best_TACL-{args.tag}.pt" for i in range(args.total_tasks)], scaling_coef=args.scaling_coef)
# print(final_model)
for task_idx, loader in enumerate(test_all_tasks):
    print(task_idx)
    test(args, final_model, loader)

torch.save(final_model, f"{args.output_dir}/resultant_model_{args.dataset}.pt")