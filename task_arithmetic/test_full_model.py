import torch
import argparse
from tqdm import tqdm
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from transformers import AutoImageProcessor
from apply_ta import get_model
from data import *

parser = argparse.ArgumentParser()

parser.add_argument('-e', '--epochs', type=int, default=50)
parser.add_argument('-bs', '--batch-size', type=int, default=16)
parser.add_argument('-lr', '--lr', type=float, default=5e-6)
parser.add_argument('-wd', '--weight-decay', type=float, default=1e-6)
parser.add_argument('-nw','--num-workers', type=int, default=2)
parser.add_argument('--test-interval', type=int, default=1)
parser.add_argument('--device', type=str, default="cuda:0")
parser.add_argument('--model', type=str, default="resnet18")
# parser.add_argument('--seed', type=int, default=42)

parser.add_argument('-d', '--dataset', type=str, default='oxfordpet')
parser.add_argument('-ddir', '--data-dir', type=str, default='../data')
parser.add_argument('-odir', '--output-dir', type=str, default='./output')
parser.add_argument('-midir', '--model-input-dir', type=str, default='./data')
parser.add_argument('-t', '--tasknum', type=int)
parser.add_argument('-tot', '--total-tasks', type=int)
parser.add_argument('-scoef', '--scaling-coef', type=float, default=0.25)
parser.add_argument('--tag', type=str, default="")

parser.add_argument('-nc', '--num-classes', type=int, default=None)
args = parser.parse_args()

if args.tag != "":
    args.tag = "-" + args.tag

args.output_dir = f"{args.output_dir}/{args.model}_{args.dataset}"

if args.num_classes == None:
    if args.dataset == 'mnist' or args.dataset == 'cifar10':
        args.num_classes = 10

def accuracy(true,pred):
    true = np.array(true)
    pred = np.array(pred)
    acc = np.sum((true == pred).astype(np.float32)) / len(true)
    return acc * 100

def test_model(model,test_loader):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
            test_loss = []
            test_preds = []
            test_labels = []
            for batch in tqdm(test_loader):
                imgs = torch.Tensor(batch[0]).to(device)
                labels = torch.Tensor(batch[1]).to(device)
                bbone_params, scores = model(imgs)
                loss = criterion(scores, labels)
                test_loss.append(loss.detach().cpu().numpy())
                test_labels.append(batch[1])
                test_preds.append(scores.argmax(dim=-1))
            loss = sum(test_loss)/len(test_loss)
            acc = accuracy(torch.concat(test_labels, dim=0).cpu(),torch.concat(test_preds, dim=0).cpu())
            print(f"\tTest:\tLoss - {round(loss, 3)}",'\t',f"Accuracy - {round(acc,3)}")
            torch.save(final_model, f"{args.output_dir}/resultant_model_{args.dataset}{args.tag}.pt")
            return loss, acc

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")

transform = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize(mean=dataset_stats_dict[args.dataset]['mean'], std=dataset_stats_dict[args.dataset]['std'])
                        ])
        
if args.dataset == 'cifar10':
    testset = datasets.CIFAR10(
            root=args.data_dir, train=False, download=True,
            transform=transform
        )

elif args.dataset == 'mnist':
    testset = datasets.MNIST(
            root=args.data_dir, train=False, download=True,
            transform=transform
        )

test_loader = DataLoader(testset,args.batch_size,num_workers=args.num_workers,shuffle=False)
# final_model = get_model(args, f"{args.output_dir}/resnet18/resnet18-pretrained.pth", list_of_task_checkpoints=[f"{args.model_input_dir}/resnet18_task_{i}_best_TACL.pt" for i in range(args.total_tasks)], scaling_coef=args.scaling_coefficient)

pretrained_path = '/content/drive/MyDrive/DL 566 Project/colab_output/resnet18/resnet18-pretrained.pth'
pretrained_model = torch.load(pretrained_path)
final_model = get_model(args, pretrained_path, list_of_task_checkpoints=[f"{args.model_input_dir}/{args.model}_task_{i}_best_TACL{args.tag}.pt" for i in range(args.total_tasks)], scaling_coef=args.scaling_coef)
test_model(final_model,test_loader)