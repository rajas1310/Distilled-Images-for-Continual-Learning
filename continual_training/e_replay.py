from torchvision import datasets, transforms
from avalanche.benchmarks.generators import nc_benchmark, ni_benchmark
import torch
from transformers import ViTModel
import torch.nn as nn

class ResNet(nn.Module):
    def __init__(self, args, model_name='resnet18'):
        super().__init__()
        self.device = args.device
        self.model_name = model_name

        self.net = torch.hub.load('pytorch/vision:v0.10.0', self.model_name, pretrained=True)
