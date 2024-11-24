from net import ResNet, test, fit
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from torchvision import transforms
import torch
from logger_utils import Logger
import sys, os, time
import random

from data import *

import argparse

parser = argparse.ArgumentParser()

parser.add_argument('-e', '--epochs', type=int, default=10)
parser.add_argument('-ns', '--num-samples', type=int, default=10)
parser.add_argument('-bs', '--batch-size', type=int, default=4)
parser.add_argument('-lr', '--lr', type=float, default=5e-6)
parser.add_argument('-wd', '--weight-decay', type=float, default=1e-6)
parser.add_argument('-p', '--patience', type=int, default=10)
parser.add_argument('-nw','--num-workers', type=int, default=2)
parser.add_argument('--test-interval', type=int, default=1)
parser.add_argument('--device', type=str, default="cuda:0")
parser.add_argument('--model', type=str, default="resnet18")
parser.add_argument('--weights', type=str)
parser.add_argument('--syn', action="store_true")
parser.add_argument('-expt', '--expt-type', type=str, default=None) # TODO : can also be "FShotTuning"
parser.add_argument('--tag', type=str, default="")
parser.add_argument('-t', '--tasknum', type=str, default="")

# parser.add_argument('--seed', type=int, default=42)

parser.add_argument('-d', '--dataset', type=str, default='cifar10')
parser.add_argument('-ddir', '--data-dir', type=str, default='./data')
parser.add_argument('-odir', '--output-dir', type=str, default='./output')
parser.add_argument('-nc', '--num-classes', type=int, default=None)
args = parser.parse_args()

if args.tag != "":
    args.tag = "-" + args.tag


args.output_dir = f"{args.output_dir}/{args.model}_{args.dataset}"

if args.num_classes == None:
    if args.dataset == 'mnist' or args.dataset == 'cifar10':
        args.num_classes = 10

model = torch.load(args.weights)


dataset_stats_dict = {  'cifar10' : {   'mean' : (0.491, 0.482, 0.447),
                                        'std' : (0.202, 0.199, 0.201)
                                    },
                        'mnist' : { 'mean' : (0.131,),
                                    'std' : (0.308,)
                                    },
                        'imagenet' : {  'mean' : (0.485, 0.456, 0.406),
                                        'std' : (0.229, 0.224, 0.225)
                                    }
                    }

transforms_list = [transforms.ToTensor(),
                   #transforms.RandomHorizontalFlip(),
                   transforms.Normalize(mean=dataset_stats_dict[args.dataset]['mean'], std=dataset_stats_dict[args.dataset]['std'])
                  ]
if args.dataset == "mnist":
    transforms_list.append(transforms.Lambda(lambda x: x.repeat(3, 1, 1)))

transform = transforms.Compose(transforms_list)
if args.dataset == "cifar10":
    trainset = datasets.CIFAR10(
            root="/content/data/", train=True, download=True,
            transform=transform
        )
    
    testset = datasets.CIFAR10(
            root=args.data_dir, train=False, download=True,
            transform=transform
        )
    
elif args.dataset == "mnist":
    trainset = datasets.MNIST(
            root="/content/data/", train=True, download=True,
            transform=transform
        )
    testset = datasets.MNIST(
            root=args.data_dir, train=False, download=True,
            transform=transform
        )

num_samples_per_class = args.num_samples
sampled_images = []

class_counts = {}  # Keep track of how many samples we've taken from each class

for i in range(len(trainset)):
    image, label = trainset[i]
    # if args.dataset == 'mnist':
    #     image = torch.squeeze(torch.stack((image,image,image), dim=1))
        
    if label not in class_counts:
        class_counts[label] = 0

    if class_counts[label] < num_samples_per_class:
        sampled_images.append((image, label))
        class_counts[label] += 1

    if all(count == num_samples_per_class for count in class_counts.values()):
        break

print(f"Sampled {len(sampled_images)} images.")

trainloader = torch.utils.data.DataLoader(
        sampled_images, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers
    )
testloader = torch.utils.data.DataLoader(
        testset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers
    )
test(args, model, testloader)
fit(args, model, trainloader, testloader)