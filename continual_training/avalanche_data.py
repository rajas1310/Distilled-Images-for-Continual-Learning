from torchvision import datasets, transforms
# from avalanche.benchmarks.generators import nc_benchmark, ni_benchmark
from avalanche.benchmarks.scenarios.deprecated.generators import nc_benchmark
import torch
import torch.nn as nn
import os
from torchvision.datasets import DatasetFolder
from torchvision.io import read_image
from torch.utils.data import Subset
import random
class ResNet(nn.Module):
    def __init__(self, args, model_name='resnet18'):
        super().__init__()
        self.device = args.device
        self.model_name = model_name

        self.net = torch.hub.load('pytorch/vision:v0.10.0', self.model_name, pretrained=True)
        self.net.fc = nn.Linear(512, args.num_classes)
        self.net.to(self.device)
        # self.linear.to(self.device)

    def forward(self, x):
        x = self.net(x)
        return x

    def print_trainable_parameters(self):
        trainable_params = 0
        all_param = 0
        for _, param in self.net.named_parameters():
            all_param += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
        print(
            f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.2f}"
        )

class CustomSyntheticDataset():
    def __init__(self, args):
        self.args = args

        if args.dataset == 'cifar10':
            image_mean = (0.491, 0.482, 0.447)
            image_std = (0.202, 0.199, 0.201)
        elif args.dataset == 'mnist':
            image_mean = (0.131,)
            image_std = (0.308,)
        elif args.dataset == 'imagenet':
            image_mean = (0.485, 0.456, 0.406)
            image_std = (0.229, 0.224, 0.225)

        self.transform =  transforms.Compose([
                                transforms.Normalize(mean=image_mean, std=image_std)
                                ]) 
        self.set_data_variables()
    
    def load_image_as_tensor(self, path):
        return read_image(path).float() / 255.0  # Normalize pixel values to [0, 1]

    def set_data_variables(self):
        if self.args.dataset == 'cifar10':
            self.task_dict = {0:2, 1:2, 2:2, 3:2, 4:2}

            self.train_datasets = []
            self.test_datasets = []

            train_ratio = 0.8

            for stage in range(len(self.task_dict)):
                stage_dir = os.path.join(self.args.data_dir, f"task{stage}")
                
                full_dataset = DatasetFolder(
                root=stage_dir,
                loader=self.load_image_as_tensor,
                extensions=("png"),
                transform=self.transform
                )
                class_indices = {}
                for idx, (_, label) in enumerate(full_dataset):
                    if label not in class_indices:
                        class_indices[label] = []
                    class_indices[label].append(idx)
                train_indices = []
                test_indices = []
                for label, indices in class_indices.items():
                    random.shuffle(indices)
                    split_point = int(len(indices) * train_ratio)
                    train_indices.extend(indices[:split_point])
                    test_indices.extend(indices[split_point:])
                trainset = Subset(full_dataset, train_indices)
                testset = Subset(full_dataset, test_indices)

                self.train_datasets.append(trainset)
                self.test_datasets.append(testset)
                
            self.trainset = torch.utils.data.ConcatDataset(self.train_datasets)
            self.testset = torch.utils.data.ConcatDataset(self.test_datasets)

        elif self.args.dataset == 'mnist':
            pass

        elif self.args.dataset == 'imagenet':
            pass
        


    def get_scenario(self):
        scenario = nc_benchmark(
            self.train_datasets, self.test_datasets, n_experiences=len(self.task_dict), per_exp_classes=self.task_dict, shuffle=False, task_labels=True
        )
        print("Trainset : ", len(self.trainset), "Testset : ", len(self.testset))
        return scenario

class CustomOriginalDataset():
    def __init__(self, args):
        self.args = args

        if args.dataset == 'cifar10':
            image_mean = (0.491, 0.482, 0.447)
            image_std = (0.202, 0.199, 0.201)
        elif args.dataset == 'mnist':
            image_mean = (0.131,)
            image_std = (0.308,)
        elif args.dataset == 'imagenet':
            image_mean = (0.485, 0.456, 0.406)
            image_std = (0.229, 0.224, 0.225)

        self.train_transform =  transforms.Compose([
                                transforms.RandomHorizontalFlip(),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=image_mean, std=image_std)
                                ]) 
        self.test_transform = transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize(mean=image_mean, std=image_std)
                                ])
        self.set_data_variables()

    def set_data_variables(self):
        if self.args.dataset == 'cifar10':
            self.trainset = datasets.CIFAR10(root=self.args.data_dir, train=True, download=True,
                                    transform=self.train_transform)
            self.testset = datasets.CIFAR10(root=self.args.data_dir, train=False, download=True,
                                   transform=self.test_transform)
            # self.trainset.targets = self.trainset._labels
            # self.testset.targets = self.testset._labels
            self.task_dict = {0:2, 1:2, 2:2, 3:2, 4:2} # task_id : num_classes

        elif self.args.dataset == 'mnist':
            self.trainset = datasets.MNIST(root=self.args.data_dir, train=True, download=True,
                                    transform=self.train_transform)
            self.testset = datasets.MNIST(root=self.args.data_dir, train=False, download=True,
                                   transform=self.test_transform)
            # self.trainset.targets = self.trainset._labels
            # self.testset.targets = self.testset._labels
            self.task_dict = {0:2, 1:2, 2:2, 3:2, 4:2} # task_id : num_classes

        elif self.args.dataset == 'imagenet':
            self.trainset = datasets.ImageNet(root=self.args.data_dir, split='train', transform=self.train_transform)
            self.testset = datasets.ImageNet(root=self.args.data_dir, split='val', transform=self.test_transform)
            pass
        


    def get_scenario(self):
        scenario = nc_benchmark(
            self.trainset, self.testset, n_experiences=len(self.task_dict), per_exp_classes=self.task_dict, shuffle=False, task_labels=True
        )
        print("Trainset : ", len(self.trainset), "Testset : ", len(self.testset))
        return scenario
