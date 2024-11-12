import torch
import torch.nn as nn
from torch.utils.data import ConcatDataset, Dataset

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