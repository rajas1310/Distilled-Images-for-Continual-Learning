import os
import sys
import time
import random
import argparse
import numpy as np

import torch
import torch.nn as nn
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchvision.utils import save_image, make_grid

task_dict = {'cifar10' : { 0:['airplane','automobile'],
                            1:['bird','cat'],
                            2:['deer','dog'],
                            3:['frog','horse'],
                            4:['ship','truck'] },
            
            'mnist' : { x:[str(x), str(x+1)] for x in range(0,10,2)},
            
            'imagenet' : {  0 : [x for x in range(1,11)] , 1 : [x for x in range(11, 21)],
                            2 : [x for x in range(21,31)], 3 : [x for x in range(31, 41)],
                            4 : [x for x in range(41,51)], 5 : [x for x in range(51, 61)],
                            6 : [x for x in range(61,71)], 7 : [x for x in range(71, 81)],
                            8 : [x for x in range(81,91)], 9 : [x for x in range(91, 103)]
                        }

            }

label2int = {'cifar10' : { 0:'airplane',
                            1:'automobile',
                            2:'bird',
                            3:'cat',
                            4:'deer',
                            5:'dog',
                            6:'frog',
                            7:'horse',
                            8:'ship',
                            9:'truck' },
            
            'mnist' : { x:str(x) for x in range(10)},
            
            'imagenet' : {  0 : [x for x in range(1,11)] , 1 : [x for x in range(11, 21)],
                            2 : [x for x in range(21,31)], 3 : [x for x in range(31, 41)],
                            4 : [x for x in range(41,51)], 5 : [x for x in range(51, 61)],
                            6 : [x for x in range(61,71)], 7 : [x for x in range(71, 81)],
                            8 : [x for x in range(81,91)], 9 : [x for x in range(91, 103)]
                        }

            }

def load_dataset(args):
    # Obtain dataloader
    transform_train = transforms.Compose([
        transforms.Resize((args.size, args.size)),
        transforms.ToTensor(),
    ])
    if args.dataset == 'cifar10':
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.491, 0.482, 0.447), (0.202, 0.199, 0.201))
        ])
        trainset = datasets.CIFAR10(root=args.data_dir, train=True, download=True,
                                    transform=transform_train)
        testset = datasets.CIFAR10(root=args.data_dir, train=False, download=True,
                                   transform=transform_test)
    elif args.dataset == 'cifar100':
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.507, 0.486, 0.441), (0.267, 0.256, 0.276))
        ])
        trainset = datasets.CIFAR100(root=args.data_dir, train=True, download=False,
                                    transform=transform_train)
        testset = datasets.CIFAR100(root=args.data_dir, train=False, download=False,
                                   transform=transform_test)  
    elif args.dataset == 'imagenet':
        transform_test = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        trainset = datasets.ImageFolder(root=args.data_dir + "/train", 
                                        transform=transform_train)
        testset = datasets.ImageFolder(root=args.data_dir + "/val", 
                                       transform=transform_train)
    elif args.dataset == 'tiny_imagenet':
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        trainset = datasets.ImageFolder(root=args.data_dir + "/train", 
                                        transform=transform_train)
        testset = datasets.ImageFolder(root=args.data_dir + "/val", 
                                       transform=transform_train)
    
    label_list = task_dict[args.dataset][args.task]
    trainset = [(img,label) for (img, label) in trainset if label2int[args.dataset][label] in label_list]
    testset = [(img,label) for (img, label) in testset if label2int[args.dataset][label] in label_list]

    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, drop_last=False
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers
    )

    return trainloader, testloader


def load_syn_dataset(args):
    # Obtain dataloader
    if args.dataset == 'syn_cifar10':
        transform_train = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize([0.5070751592371323, 0.48654887331495095, 0.4409178433670343], [0.2673342858792401, 0.2564384629170883, 0.27615047132568404])
        ])
        trainset = datasets.ImageFolder(root=args.data_dir, 
                                        transform=transform_train)

    elif args.dataset == 'syn_imagenet':
        transform_train = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        trainset = datasets.ImageFolder(root=args.data_dir, 
                                        transform=transform_train)
       

    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers
    )
    return trainloader


def gen_label_list(args):
    # obtain label-prompt list
    with open(args.label_file_path, "r") as f:
        lines = f.readlines()

    labels = []
    for line in lines:
        line = line.strip()
        label = line.split('\t')[0]
        labels.append(label)
    
    return labels
