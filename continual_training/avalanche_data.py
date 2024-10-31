from torchvision import datasets, transforms
# from avalanche.benchmarks.generators import nc_benchmark, ni_benchmark
from avalanche.benchmarks.scenarios.deprecated.generators import nc_benchmark
import torch
import torch.nn as nn
import os
from torch.utils.data import Subset
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

        self.train_transform =  transforms.Compose([
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=image_mean, std=image_std)
                    ]) 
        self.test_transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(mean=image_mean, std=image_std)
                    ])
        self.task_dict = {
            'cifar10': {
                0: ['airplane', 'automobile'],
                1: ['bird', 'cat'],
                2: ['deer', 'dog'],
                3: ['frog', 'horse'],
                4: ['ship', 'truck']
            }
        }
        self.set_data_variables()

    def set_data_variables(self):
        if self.args.dataset == 'cifar10':

            self.trainset = []
            self.testset = []

            for stage in range(len(self.task_dict['cifar10'])):
                stage_dir = os.path.join(self.args.data_dir, f"task{stage}")
                
                train_dataset = datasets.ImageFolder(
                    root=stage_dir,
                    transform=self.train_transform
                )
                self.trainset.append(train_dataset)

            cifar10_class_to_idx = {
                'airplane': 0, 'automobile': 1, 'bird': 2, 'cat': 3,
                'deer': 4, 'dog': 5, 'frog': 6, 'horse': 7, 'ship': 8, 'truck': 9
            }

            full_test_dataset = datasets.CIFAR10(
                root=self.args.data_dir, train=False, download=True,
                transform=self.test_transform
            )
            for stage, class_names in self.task_dict['cifar10'].items():
                class_indices = [cifar10_class_to_idx[class_name] for class_name in class_names]
                test_indices = [i for i, (_, label) in enumerate(full_test_dataset) if label in class_indices]
                remapped_labels = torch.tensor([class_indices.index(full_test_dataset.targets[i]) for i in test_indices])
                test_subset = Subset(full_test_dataset, test_indices)
                test_subset.targets = remapped_labels
                print(f"Stage {stage}, Classes: {class_names}, Unique Labels in Test Subset: {set(test_subset.targets.tolist())}")
                self.testset.append(test_subset)

        elif self.args.dataset == 'mnist':
            pass

        elif self.args.dataset == 'imagenet':
            pass


    def get_scenario(self):
        if self.args.dataset == 'cifar10':
            print("Trainset : ", len(self.trainset), "Testset : ", len(self.testset))
            scenario = nc_benchmark(
                self.trainset, self.testset, 
                n_experiences=len(self.task_dict['cifar10']),
                one_dataset_per_exp = True,
                class_ids_from_zero_in_each_exp = True,
                # per_exp_classes={i: len(classes) for i, classes in self.task_dict['cifar10'].items()},
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
