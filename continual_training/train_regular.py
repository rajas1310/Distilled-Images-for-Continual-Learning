import torchvision.datasets as datasets
import torch
from avalanche_data import ResNet, CustomSyntheticDataset, CustomOriginalDataset, ImageDataset
from avalanche.training.supervised import JointTraining
from avalanche.logging import TextLogger, InteractiveLogger, WandBLogger
from avalanche.training.plugins import EvaluationPlugin, EarlyStoppingPlugin
from avalanche.evaluation.metrics import forgetting_metrics, accuracy_metrics,\
    loss_metrics, timing_metrics, cpu_usage_metrics, StreamConfusionMatrix,\
    disk_usage_metrics, gpu_usage_metrics
import numpy as np
from torch.utils.data import Dataset
import torch.nn as nn
from datetime import datetime
from logger_utils import Logger
import argparse
from torchvision import datasets, transforms

from tqdm import tqdm

import sys, os


parser = argparse.ArgumentParser()

parser.add_argument('-e', '--epochs', type=int, default=50)
parser.add_argument('--eval-epochs', type=int, default=0)
parser.add_argument('-bs', '--batch-size', type=int, default=16)
parser.add_argument('-lr', '--lr', type=float, default=5e-6)
parser.add_argument('-wd', '--weight-decay', type=float, default=1e-6)
parser.add_argument('-p', '--patience', type=int, default=10)
parser.add_argument('-nw','--num-workers', type=int, default=2)
parser.add_argument('--device', type=str, default="cuda:0")
parser.add_argument('--test-interval', type=int, default=1)

# parser.add_argument('--seed', type=int, default=42)

parser.add_argument('-d', '--dataset', type=str, default='cifar10')
parser.add_argument('-ddir', '--data-dir', type=str, default='../../data')
parser.add_argument('-odir', '--output-dir', type=str, default='./output')
parser.add_argument('-m', '--model', type=str, default='resnet18')
parser.add_argument('-nc', '--num-classes', type=int, default=None)
args = parser.parse_args()


if args.num_classes == None:
    if args.dataset == 'mnist' or args.dataset == 'cifar10':
        args.num_classes = 10
    elif args.dataset == 'imagenet':
        args.num_classes = 1000
    elif args.dataset == 'stanfordcars':
        args.num_classes = 196

print(args, '\n')

args.output_dir = f'{args.output_dir}/{args.model}'

if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir, exist_ok=True)
    
# sys.stdout = Logger(os.path.join(args.output_dir, 'logs-{}-{}-{}.txt'.format(args.dataset, args.strategy, args.epochs)))
timestamp = datetime.now().strftime("%d-%m-%Y %H:%M:%S")
log_filename = os.path.join(args.output_dir, 'logs-{}-Offline-{}-{}.txt'.format(args.dataset,  args.epochs, timestamp))

sys.stdout = Logger(log_filename)


model = ResNet(args)
# scenario = CustomSyntheticDataset(args).get_scenario()
trainset = ImageDataset(args, split="train")
testset = datasets.CIFAR10(root=args.data_dir, train=False, download=True,
                                   transform=transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize(mean= (0.491, 0.482, 0.447), std=(0.202, 0.199, 0.201))
                                ]))

print("Trainset samples:", len(trainset), "Test samples:", len(testset))

train_loader = torch.utils.data.DataLoader(
        trainset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers
    )
test_loader = torch.utils.data.DataLoader(
        testset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers
    )

optimizer = torch.optim.Adam(
            params=model.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )
criterion = nn.CrossEntropyLoss()

def accuracy(true, pred):
        true = np.array(true)
        pred = np.array(pred)
        acc = np.sum((true == pred).astype(np.float32)) / len(true)
        return acc * 100

def test(test_loader):
    criterion = nn.CrossEntropyLoss()
    model.eval()
    with torch.no_grad():
        test_loss = []
        test_preds = []
        test_labels = []
        for batch in tqdm(test_loader):
            imgs = torch.Tensor(batch[0]).to(args.device)
            labels = torch.Tensor(batch[1]).to(args.device)
            scores = model(imgs)
            loss = criterion(scores, labels)
            test_loss.append(loss.detach().cpu().numpy())
            test_labels.append(batch[1])
            test_preds.append(scores.argmax(dim=-1))
        loss = sum(test_loss)/len(test_loss)
        acc = accuracy(torch.concat(test_labels, dim=0).cpu(),torch.concat(test_preds, dim=0).cpu())
        print(f"\tTest:\tLoss - {round(loss, 3)}",'\t',f"Accuracy - {round(acc,3)}")
        
        return loss, acc

model.train()
best_test_acc = -np.inf

for epoch in range(args.epochs):
    print(f"{epoch}/{args.epochs-1} epochs")
    train_loss = []
    train_preds = []
    train_labels = []
    for batch in tqdm(train_loader):
        imgs = torch.Tensor(batch[0]).to(args.device)
        labels = torch.Tensor(batch[1]).to(args.device)
        scores = model(imgs)
        loss = criterion(scores, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss.append(loss.detach().cpu().numpy())
        train_labels.append(batch[1])
        train_preds.append(scores.argmax(dim=-1))
    loss = sum(train_loss)/len(train_loss)
    acc = accuracy(torch.concat(train_labels, dim=0).cpu(),torch.concat(train_preds, dim=0).cpu())
    print(f"\tTrain\tLoss - {round(loss, 3)}",'\t',f"Accuracy - {round(acc, 3)}")

    if (epoch+1) % args.test_interval == 0:
        test_loss, test_acc = test(test_loader)
        if test_acc > best_test_acc:
            patient_epochs = 0
            best_test_acc = test_acc
            print(f"\tCurrent best epoch : {epoch} \t Best test acc. : {round(best_test_acc,3)}")
            torch.save(model.state_dict(), f"{args.output_dir}/{args.model}_best.pt")
    else:
        patient_epochs += 1
    
    if patient_epochs == args.patience:
        print("INFO: Accuracy has not increased in the last {} epochs.".format(args.patience))
        print("INFO: Stopping the run and saving the best weights.")
        break
    print("--"*100)