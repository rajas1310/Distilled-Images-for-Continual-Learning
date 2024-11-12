from torchvision import datasets, transforms
# from avalanche.benchmarks.generators import nc_benchmark, ni_benchmark
from avalanche.benchmarks.scenarios.deprecated.generators import nc_benchmark
import torch
# from transformers import ViTModel
import torch.nn as nn
from torch.utils.data import ConcatDataset, Dataset
import glob
from pathlib import Path
from PIL import Image

task_dict = {'cifar10' : {  0 : ['airplane', 'automobile'],
                            1 : ['bird', 'cat'],
                            2 : ['deer', 'dog'],
                            3 : ['frog', 'horse'],
                            4 : ['ship', 'truck']
                          },

             'mnist' : {    0 : [0, 1],
                            1 : [2, 3],
                            2 : [4, 5],
                            3 : [6, 7],
                            4 : [8, 9]
                        }
            }

label2int = {'cifar10' : {'airplane' : 0, 'automobile' : 1,
                          'bird' : 2, 'cat' : 3,
                          'deer' : 4, 'dog' : 5,
                          'frog' : 6, 'horse' : 7,
                          'ship' : 8, 'truck' : 8
                          }
            
            }

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

class ImageDataset(Dataset):
    def __init__(self, args, image_list, label_list, split='train'):
        super().__init__()
        self.args = args
        self.data_stats_dict = dataset_stats_dict[args.dataset]
        self.image_list = image_list #glob.glob(f'{args.data_dir}/task*/*/*')
        self.label_list = label_list #[int(Path(x).parent.stem) for x in self.image_list]
        """
        if args.dataset == 'cifar10':
            image_mean = (0.491, 0.482, 0.447)
            image_std = (0.202, 0.199, 0.201)
        elif args.dataset == 'mnist':
            image_mean = (0.131,)
            image_std = (0.308,)
        elif args.dataset == 'imagenet':
            image_mean = (0.485, 0.456, 0.406)
            image_std = (0.229, 0.224, 0.225)
        """
        if split == 'train':
            self.transform =  transforms.Compose([
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=self.data_stats_dict['mean'], std=self.data_stats_dict['std'])
                        ]) 
        elif split == 'test':
            self.transform = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize(mean=self.data_stats_dict['mean'], std=self.data_stats_dict['std'])
                        ])

    def __len__(self):
        return len(self.label_list)

    def __getitem__(self, idx):
        img = self.transform(Image.open(self.image_list[idx]).convert('RGB'))
        label = self.label_list[idx]
        return img, label
    
class SyntheticTrainDataset():
    def __init__(self, args, task_dict=task_dict):
        self.args = args
        # self.img_processor = img_processor
        self.task_dict = task_dict[args.dataset]
        self.label2int = self.get_label2int()
        self.train_imgs, self.train_labels = [],[]
        # self.test_imgs, self.test_labels = [],[]
        self.get_lists()

    def get_label2int(self):
        label2int = []
        for task in self.task_dict.values():
            for clas in task:
                label2int.append(clas)

        keys = [x for x in range(len(label2int))]
        label2int = dict(zip(label2int, keys))
        return label2int

    def get_list(self):
        # trainset
        for image_path in glob.glob(f'{self.args.data_dir}/Task*/*/.png'):
            self.train_imgs.append(image_path)
            label = int(Path(image_path).parent.stem)
            self.train_labels.append(label)

    def get_dataset(self):
        print(f"INFO : Loading {self.args.dataset} TRAIN data for TASK {self.args.tasknum} ... ")
        print("CLASSES : ", self.task_dict[self.args.tasknum])
        return ImageDataset(self.train_imgs, self.train_labels, 'train')


class TaskwiseTestDataset():
    def __init__(self, args, task_dict=task_dict, include_previous_tasks = False):
        self.args = args
        # self.img_processor = img_processor
        self.include_previous_tasks = include_previous_tasks
        self.task_dict = task_dict[args.dataset]
        self.label2int = self.get_label2int()
        self.test_imgs, self.test_labels = [],[]
        self.get_lists()

    def get_label2int(self):
        label2int = []
        for task in self.task_dict.values():
            for clas in task:
                label2int.append(clas)

        keys = [x for x in range(len(label2int))]
        label2int = dict(zip(label2int, keys))
        return label2int
    
    def get_list(self):
        self.transform = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize(mean=self.data_stats_dict['mean'], std=self.data_stats_dict['std'])
                        ])
        
        if self.args.dataset == 'cifar10':
            self.testset = datasets.CIFAR10(
                    root=self.args.data_dir, train=False, download=True,
                    transform=self.transform
                )
            
            imgs = self.testset.data
            labels = self.testset.targets

            if self.include_previous_tasks:
                valid_labels = [label2int['cifar10'][item] for x in range(self.args.tasknum + 1) for item in self.task_dict[x]]
            else:
                valid_labels = self.task_dict[self.args.tasknum]

        elif self.args.dataset == 'mnist':
            self.testset = datasets.MNIST(
                    root=self.args.data_dir, train=False, download=True,
                )
            imgs = self.testset.test_data
            labels = self.testset.test_labels

            if self.include_previous_tasks:
                valid_labels = [item for x in range(self.args.tasknum + 1) for item in self.task_dict[x]]
            else:
                valid_labels = self.task_dict[self.args.tasknum]

        for idx,label in enumerate(labels):
            if label in valid_labels:
                self.test_imgs.append(imgs[idx])
                self.test_labels.append(labels[idx])

    def get_dataset(self):
        print(f"INFO : Loading {self.args.dataset} TEST data for TASK {self.args.tasknum} ... ")
        print("CLASSES : ", self.task_dict[self.args.tasknum])
        return ImageDataset(self.test_imgs, self.test_labels, 'test')

        

            

    


