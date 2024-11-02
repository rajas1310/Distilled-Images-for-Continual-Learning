from torchvision import datasets, transforms
# from avalanche.benchmarks.generators import nc_benchmark, ni_benchmark
from avalanche.benchmarks.scenarios.deprecated.generators import nc_benchmark
import torch
from transformers import ViTModel
import torch.nn as nn
from torch.utils.data import ConcatDataset, Dataset
import glob
from pathlib import Path
from PIL import Image

class ResNet(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.device = args.device
        self.model_name = args.model

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

class ImageDataset(Dataset):
    def __init__(self, args, split='train'):
        super().__init__()
        self.args = args
        self.image_list = glob.glob(f'{args.data_dir}/*/*/*')
        self.label_list = [int(Path(x).parent.stem) for x in self.image_list]

        if args.dataset == 'cifar10':
            image_mean = (0.491, 0.482, 0.447)
            image_std = (0.202, 0.199, 0.201)
        elif args.dataset == 'mnist':
            image_mean = (0.131,)
            image_std = (0.308,)
        elif args.dataset == 'imagenet':
            image_mean = (0.485, 0.456, 0.406)
            image_std = (0.229, 0.224, 0.225)

        if split == 'train':
            self.transform =  transforms.Compose([
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=image_mean, std=image_std)
                        ]) 
        elif split == 'test':
            self.transform = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize(mean=image_mean, std=image_std)
                        ])

    def __len__(self):
        return len(self.label_list)

    def __getitem__(self, idx):
        img = self.transform(Image.open(self.image_list[idx]).convert('RGB'))
        label = self.label_list[idx]
        return img, label

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
            trainsets = []
            for task_dir in glob.glob(f'{self.args.data_dir}/task*'):
                trainsets.append(datasets.ImageFolder(
                    root=task_dir,
                    transform=self.train_transform
                ))
            self.trainset = ConcatDataset(trainsets)

            self.testset = datasets.CIFAR10(
                root="../../cifar10", train=False, download=True,
                transform=self.test_transform
            )

            self.task_dict = {0:2, 1:2, 2:2, 3:2, 4:2}

        elif self.args.dataset == 'mnist':
            pass

        elif self.args.dataset == 'imagenet':
            pass


    def get_scenario(self):
        if self.args.dataset == 'cifar10':
            print("Trainset : ", len(self.trainset), "Testset : ", len(self.testset))
            scenario = nc_benchmark(
                self.trainset, self.testset, 
                n_experiences=len(self.task_dict),
                per_exp_classes=self.task_dict,
                shuffle=False,
                task_labels=True
            )
            return scenario
        
        elif self.args.dataset == 'mnist':
            pass

        elif self.args.dataset == 'imagenet':
            pass


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
