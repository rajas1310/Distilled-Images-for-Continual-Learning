from re import X
import torch
import torch.nn as nn
from torch.utils.data import ConcatDataset, Dataset
from tqdm import tqdm
import numpy as np

class ResNet(nn.Module):
    def __init__(self, num_classes, device ,model='resnet18'):
        super().__init__()
        self.device = device
        self.model_name = model

        self.net = torch.hub.load('pytorch/vision:v0.10.0', self.model_name, pretrained=True)
        self.net.fc = nn.Identity()
        self.linear = nn.Linear(512, num_classes)

        self.net.to(self.device)
        self.linear.to(self.device)

        

    def forward(self, x):
        x = self.net(x)

        return x, self.linear(x)

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
    
    def accuracy(self, true, pred):
        true = np.array(true)
        pred = np.array(pred)
        acc = np.sum((true == pred).astype(np.float32)) / len(true)
        return acc * 100


def fit(args, model, train_loader, test_loader, pretrained_model=None):
    optim = torch.optim.Adam(
        params=model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    criterion = nn.CrossEntropyLoss()
      
    if pretrained_model: # for KL-div loss
      criterion_kldiv = nn.KLDivLoss(size_average=False)
      softmax = nn.Softmax(dim=1)

    model.train()

    best_test_acc = -np.inf
    best_epoch = 0

    for epoch in range(args.epochs):
        print(f"{epoch}/{args.epochs-1} epochs   -> (Task-{args.tasknum})")
        train_loss = []
        train_preds = []
        train_labels = []
        for batch in tqdm(train_loader):
            imgs = torch.Tensor(batch[0]).to(args.device)
            labels = torch.Tensor(batch[1]).to(args.device)
            bbone_params, scores = model(imgs)

            if pretrained_model:
              pretrained_model.eval()
              bbone_params_prtn, scores_prtn = pretrained_model(imgs)
              # print(self.softmax(scores_prtn),"\n", self.softmax(scores))
              loss_kldiv = criterion_kldiv(softmax(bbone_params_prtn).log(), softmax(bbone_params))
              # print("\nKLDIV : ", loss_kldiv)
              loss = criterion(scores, labels)
              # print("Loss: ", loss_kldiv, loss)
              loss = 0.6*loss_kldiv + 0.4*loss
            else:
              loss = criterion(scores, labels)

            optim.zero_grad()
            loss.backward()
            optim.step()
            train_loss.append(loss.detach().cpu().numpy())
            train_labels.append(batch[1])
            train_preds.append(scores.argmax(dim=-1))
        loss = sum(train_loss)/len(train_loss)
        acc = model.accuracy(torch.concat(train_labels, dim=0).cpu(),torch.concat(train_preds, dim=0).cpu())
        print(f"\tTrain\tLoss - {round(loss, 3)}",'\t',f"Accuracy - {round(acc, 3)}")

        if (epoch+1) % args.test_interval == 0:
            test_loss, test_acc = test(args, model, test_loader)
            if test_acc > best_test_acc:
                patient_epochs = 0
                best_test_acc = test_acc
                best_epoch = epoch
                torch.save(model.state_dict(), f"{args.output_dir}/{args.model}_task_{args.tasknum}_best_TACL-{args.tag}.pt")
            else:
                patient_epochs += 1 * args.test_interval
            print(f"\tCurrent best epoch : {best_epoch} \t Best test acc. : {round(best_test_acc,3)}")

        if patient_epochs == args.patience:
            print("INFO: Accuracy has not increased in the last {} epochs.".format(args.patience))
            print("INFO: Stopping the run and saving the best weights.")
            break
        print("--"*100)
            

def test(args, model, test_loader):
    criterion = nn.CrossEntropyLoss()
    model.eval()
    with torch.no_grad():
        test_loss = []
        test_preds = []
        test_labels = []
        for batch in tqdm(test_loader):
            imgs = torch.Tensor(batch[0]).to(args.device)
            labels = torch.Tensor(batch[1]).to(args.device)
            bbones_params, scores = model(imgs)
            loss = criterion(scores, labels)
            test_loss.append(loss.detach().cpu().numpy())
            test_labels.append(batch[1])
            test_preds.append(scores.argmax(dim=-1))
        loss = sum(test_loss)/len(test_loss)
        acc = model.accuracy(torch.concat(test_labels, dim=0).cpu(),torch.concat(test_preds, dim=0).cpu())
        print(f"\tTest:\tLoss - {round(loss, 3)}",'\t',f"Accuracy - {round(acc,3)}")
        
        return loss, acc
