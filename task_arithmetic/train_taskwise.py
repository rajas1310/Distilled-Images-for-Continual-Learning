from net import ResNet, test, fit
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from torchvision import transforms
from transformers import AutoImageProcessor
import torch
from logger_utils import Logger
import sys, os, time

from data import *

import argparse

parser = argparse.ArgumentParser()

parser.add_argument('-e', '--epochs', type=int, default=50)
parser.add_argument('-bs', '--batch-size', type=int, default=16)
parser.add_argument('-lr', '--lr', type=float, default=5e-6)
parser.add_argument('-wd', '--weight-decay', type=float, default=1e-6)
parser.add_argument('-p', '--patience', type=int, default=10)
parser.add_argument('-nw','--num-workers', type=int, default=2)
parser.add_argument('--test-interval', type=int, default=1)
parser.add_argument('--device', type=str, default="cuda:0")
parser.add_argument('--model', type=str, default="resnet18")
parser.add_argument('--syn', action="store_true")
parser.add_argument('-expt', '--expt-type', type=str, default=None) # TODO : can also be "FShotTuning"
parser.add_argument('--tag', type=str, default="")

# parser.add_argument('--seed', type=int, default=42)

parser.add_argument('-d', '--dataset', type=str, default='cifar10')
parser.add_argument('-ddir', '--data-dir', type=str, default='./data')
parser.add_argument('-odir', '--output-dir', type=str, default='./output')
parser.add_argument('-t', '--tasknum', type=int)
parser.add_argument('-nc', '--num-classes', type=int, default=None)
args = parser.parse_args()

if args.num_classes == None:
    if args.dataset == 'mnist' or args.dataset == 'cifar10':
        args.num_classes = 10

        
    # args.num_classes = 10 if args.data=='svhn' else( 37 if args.data=='oxfordpet' else ( 196 if args.data == 'stanfordcars' else 102))


print(args)

args.output_dir = f'{args.output_dir}/{args.model}_{args.dataset}'
if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir, exist_ok=True)

sys.stdout = Logger(os.path.join(args.output_dir, f'logs-task-{args.tasknum}-TACL-{args.tag}.txt'))

# img_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")
if args.syn == True:
    trainset = SyntheticTrainDataset(args).get_dataset()
    testset = TaskwiseOriginalDataset(args, split='test').get_dataset()
else:
    trainset, testset = TaskwiseOriginalDataset(args, split='train').get_dataset()


trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers
    )
testloader = torch.utils.data.DataLoader(
        testset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers
    )

start_time = time.time()

model = ResNet(num_classes=args.num_classes, device=args.device, model=args.model)

if args.expt_type == 'KLDivLoss':
    model_pretrained = ResNet(num_classes=args.num_classes, device=args.device, model=args.model)
    fit(args, model, trainloader, testloader, model_pretrained)
else:
    fit(args, model, trainloader, testloader)

end_time = time.time()

print("Time taken for training : ", round(end_time-start_time, 3), "secs")
