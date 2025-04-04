import torch
import torch.nn as nn

import numpy as np
from avalanche.training.supervised import Replay, AGEM, EWC  # and many more!
from avalanche.logging import TextLogger, InteractiveLogger
from avalanche.training.plugins import EvaluationPlugin
from avalanche.evaluation.metrics import forgetting_metrics, accuracy_metrics,\
    loss_metrics, timing_metrics, cpu_usage_metrics, StreamConfusionMatrix,\
    disk_usage_metrics, gpu_usage_metrics

from avalanche.checkpointing import maybe_load_checkpoint, save_checkpoint

from avalanche_data import CustomOriginalDataset, ResNet

import argparse

import sys, os
from logger_utils import Logger

parser = argparse.ArgumentParser()

parser.add_argument('-e', '--epochs', type=int, default=50)
parser.add_argument('-bs', '--batch-size', type=int, default=16)
parser.add_argument('-lr', '--lr', type=float, default=5e-6)
parser.add_argument('-wd', '--weight-decay', type=float, default=1e-6)
# parser.add_argument('-p', '--patience', type=int, default=10)
parser.add_argument('-nw','--num-workers', type=int, default=2)
# parser.add_argument('--test-interval', type=int, default=1)
parser.add_argument('--device', type=str, default="cuda:0")
# parser.add_argument('--seed', type=int, default=42)

parser.add_argument('--strategy', type=str, default="replay")

parser.add_argument('-d', '--dataset', type=str, default='cifar10')
parser.add_argument('-ddir', '--data-dir', type=str, default='../../data')
parser.add_argument('-odir', '--output-dir', type=str, default='./output')
args = parser.parse_args()

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

if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir, exist_ok=True)
    
sys.stdout = Logger(os.path.join(args.output_dir, 'logs-{}-{}-{}.txt'.format(args.dataset, args.strategy, args.epochs)))

model = ResNet(args)
scenario = CustomOriginalDataset(args).get_scenario()

text_logger = TextLogger(open(f'log_{args.dataset}_{args.strategy}.txt', 'a'))
interactive_logger = InteractiveLogger()

eval_plugin = EvaluationPlugin(
    accuracy_metrics(minibatch=False, epoch=True, experience=True, stream=True),
    loss_metrics(minibatch=False, epoch=True, experience=True, stream=True),
    # timing_metrics(epoch=True),
    # cpu_usage_metrics(experience=True),
    forgetting_metrics(experience=True, stream=True),
    # StreamConfusionMatrix(num_classes=args.num_classes, save_image=False),
    # disk_usage_metrics(minibatch=True, epoch=True, experience=True, stream=True),
    loggers=[text_logger, interactive_logger]
)

optimizer = torch.optim.Adam(
            params=model.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )
criterion = nn.CrossEntropyLoss()

if args.strategy == 'replay':
    cl_strategy = Replay(
        model, optimizer, criterion,
        train_mb_size=args.batch_size, train_epochs=args.epochs, eval_mb_size=args.batch_size, evaluator=eval_plugin, device=args.device
    )
elif args.strategy == 'agem':
    cl_strategy = AGEM(
        model, optimizer, criterion,
        train_mb_size=args.batch_size, train_epochs=args.epochs, eval_mb_size=args.batch_size, evaluator=eval_plugin, device=args.device,
        patterns_per_exp = 100
    )
elif args.strategy == 'ewc':
    cl_strategy = EWC(
        model, optimizer, criterion,
        train_mb_size=args.batch_size, train_epochs=args.epochs, eval_mb_size=args.batch_size, evaluator=eval_plugin, device=args.device
    )
else:
    ValueError()


# TRAINING LOOP
print('Starting experiment...')
results = []
for experience in scenario.train_stream:
    print("Start of experience: ", experience.current_experience)
    print("Current Classes: ", experience.classes_in_this_experience)

    res = cl_strategy.train(experience, num_workers = args.num_workers)
    print('Training completed')

    # save_checkpoint(cl_strategy, f'{args.output_dir}/checkpt_{args.dataset}_task{len(results)}_eps{args.epochs}.pkl')

    print('Computing accuracy on the whole test set')
    results.append(cl_strategy.eval(scenario.test_stream))
    print(results[-1])