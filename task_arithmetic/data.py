from torchvision import datasets, transforms
# from avalanche.benchmarks.generators import nc_benchmark, ni_benchmark
# from avalanche.benchmarks.scenarios.deprecated.generators import nc_benchmark
import torch
# from transformers import ViTModel
import torch.nn as nn
from torch.utils.data import ConcatDataset, Dataset
import glob
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import numpy as np

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
                        # transforms.RandomHorizontalFlip(),
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
        if self.args.dataset == 'mnist':
          self.image_list[idx] = torch.stack((self.image_list[idx],self.image_list[idx],self.image_list[idx]), dim=-1)

        # img = self.transform(Image.open(self.image_list[idx]).convert('RGB'))
        img = self.transform(Image.fromarray(np.array(self.image_list[idx])))
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
        self.get_list()

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
        print(f"\nINFO : Loading {self.args.dataset} Synthetic TRAIN ({len(self.train_labels)}) data for TASK {self.args.tasknum} ... ")
        print("CLASSES : ", self.task_dict[self.args.tasknum])
        return ImageDataset(self.args, self.train_imgs, self.train_labels, 'train')

class TaskwiseOriginalDataset():
    def __init__(self, args, split:str='test', task_dict=task_dict, include_previous_tasks = False):
        self.args = args
        # self.img_processor = img_processor
        self.split = split
        self.include_previous_tasks = include_previous_tasks
        self.data_stats_dict = dataset_stats_dict[args.dataset]
        self.task_dict = task_dict[args.dataset]
        self.label2int = self.get_label2int()
        self.train_imgs, self.train_labels = [],[]
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
    
    def get_lists(self):
        self.train_transform = transforms.Compose([
                                transforms.RandomHorizontalFlip(),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=self.data_stats_dict['mean'], std=self.data_stats_dict['std'])
                                ]) 
        
        self.test_transform = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize(mean=self.data_stats_dict['mean'], std=self.data_stats_dict['std'])
                        ])
        
        
        if self.args.dataset == 'cifar10':
            if self.split == 'train':
                self.trainset = datasets.CIFAR10(root=self.args.data_dir, train=True, download=True,
                                        transform=self.train_transform)
                
                train_imgs = self.trainset.data
                train_labels = self.trainset.targets

            self.testset = datasets.CIFAR10(root=self.args.data_dir, train=False, download=True,
                    transform=self.test_transform
                )
        
            test_imgs = self.testset.data
            test_labels = self.testset.targets


            valid_train_labels = [label2int['cifar10'][item] for item in self.task_dict[self.args.tasknum]]

            if self.include_previous_tasks:
                valid_test_labels = [label2int['cifar10'][item] for x in range(self.args.tasknum + 1) for item in self.task_dict[x]]
            else:
                valid_test_labels = valid_train_labels

        elif self.args.dataset == 'mnist':

            if self.split == 'train':
                self.trainset = datasets.MNIST(root=self.args.data_dir, train=True, download=True,
                                        transform=self.train_transform)
                
                train_imgs = self.trainset.test_data
                train_labels = self.trainset.test_labels

            self.testset = datasets.MNIST(
                    root=self.args.data_dir, train=False, download=True,
                )

            test_imgs = self.testset.test_data
            test_labels = self.testset.test_labels


            valid_train_labels = self.task_dict[self.args.tasknum]

            if self.include_previous_tasks:
                valid_test_labels = [item for x in range(self.args.tasknum + 1) for item in self.task_dict[x]]
            else:
                valid_test_labels = valid_train_labels

        #for train
        if self.split == 'train':
            for idx,label in tqdm(enumerate(train_labels), desc="Filtering train-samples as per task", ncols=25):
                
                if label in valid_train_labels:
                    self.train_imgs.append(train_imgs[idx])
                    self.train_labels.append(train_labels[idx])

        #for test
        for idx,label in tqdm(enumerate(test_labels), desc="Filtering test-samples as per task", ncols=25):
            if label in valid_test_labels:
                self.test_imgs.append(test_imgs[idx])
                self.test_labels.append(test_labels[idx])

    def get_dataset(self):
        if self.split == 'train':
            print(f"\nINFO : Loading {self.args.dataset} Original TRAIN ({len(self.train_labels)}) and TEST ({len(self.test_labels)}) data for TASK {self.args.tasknum} (includes_prev_tasks = {self.include_previous_tasks})... ")
            print("CLASSES : ", self.task_dict[self.args.tasknum])
            return ImageDataset(self.args, self.train_imgs, self.train_labels, 'train'), ImageDataset(self.args, self.test_imgs, self.test_labels, 'test') 
        elif self.split == 'test':
            print(f"\nINFO : Loading {self.args.dataset} Original TEST ({len(self.test_labels)}) data for TASK {self.args.tasknum} (includes_prev_tasks = {self.include_previous_tasks})... ")
            print("CLASSES : ", self.task_dict[self.args.tasknum])
            return ImageDataset(self.args, self.test_imgs, self.test_labels, 'test')
            

    


