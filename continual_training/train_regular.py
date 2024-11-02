import torchvision.datasets as datasets
import torch
from avalanche_data import ResNet, CustomSyntheticDataset, CustomOriginalDataset
from avalanche.training.supervised import JointTraining
from avalanche.logging import TextLogger, InteractiveLogger, WandBLogger
from avalanche.training.plugins import EvaluationPlugin, EarlyStoppingPlugin
from avalanche.evaluation.metrics import forgetting_metrics, accuracy_metrics,\
    loss_metrics, timing_metrics, cpu_usage_metrics, StreamConfusionMatrix,\
    disk_usage_metrics, gpu_usage_metrics
import torch.nn as nn
from datetime import datetime

import argparse

import sys, os


parser = argparse.ArgumentParser()

parser.add_argument('-e', '--epochs', type=int, default=50)
parser.add_argument('--eval-epochs', type=int, default=0)
parser.add_argument('-bs', '--batch-size', type=int, default=16)
parser.add_argument('-lr', '--lr', type=float, default=5e-6)
parser.add_argument('-wd', '--weight-decay', type=float, default=1e-6)
# parser.add_argument('-p', '--patience', type=int, default=10)
parser.add_argument('-nw','--num-workers', type=int, default=2)
# parser.add_argument('--test-interval', type=int, default=1)
parser.add_argument('--device', type=str, default="cuda:0")
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

model = ResNet(args)
scenario = CustomSyntheticDataset(args).get_scenario()

optimizer = torch.optim.Adam(
            params=model.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )
criterion = nn.CrossEntropyLoss()

loggers = []
loggers.append(TextLogger(open(log_filename, 'a')))
loggers.append(InteractiveLogger())
# loggers.append( WandBLogger(project_name="DL566-project", run_name="logs-{}-{}-{}-{}".format(args.dataset, args.strategy, args.epochs, timestamp)))

eval_plugin = EvaluationPlugin(
    accuracy_metrics(minibatch=False, epoch=True, experience=True, stream=True),
    loss_metrics(minibatch=False, epoch=True, experience=True, stream=True),
    # timing_metrics(epoch=True),
    # cpu_usage_metrics(experience=True),
    # forgetting_metrics(experience=True, stream=True),
    StreamConfusionMatrix(num_classes=args.num_classes, save_image=True),
    # disk_usage_metrics(minibatch=True, epoch=True, experience=True, stream=True),
    loggers=loggers
)

# Joint training strategy
joint_train = JointTraining(
        model,
        optimizer,
        criterion,
        train_mb_size=32,
        train_epochs=args.epochs,
        eval_mb_size=32,
        device=args.device,
        eval_every= args.eval_epochs
    )

# train and test loop
results = []
print("Starting training.")
# Differently from other avalanche strategies, you NEED to call train
# on the entire stream.
joint_train.train(scenario.train_stream)
results.append(joint_train.eval(scenario.test_stream))
