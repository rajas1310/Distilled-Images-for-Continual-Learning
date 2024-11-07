import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class FocalLoss(nn.Module):
    def __init__(self,  y_train, device):
        super(FocalLoss, self).__init__()
        self.alpha, self.gamma = self.get_classweights(y_train)

    def get_classweights(self, y_train):
        class_counts = np.bincount(y_train)
        num_classes = len(class_counts)
        total_samples = len(y_train)

        class_weights = []
        for count in class_counts:
            weight = 1 / (count / total_samples)
            class_weights.append(weight)
        return torch.FloatTensor(class_weights), num_classes

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        loss = (self.alpha[targets] * (1 - pt) ** self.gamma * ce_loss).mean()
        return loss