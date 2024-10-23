import torch

net = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
print(net)
