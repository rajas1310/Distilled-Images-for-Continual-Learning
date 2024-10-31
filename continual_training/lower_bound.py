import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

class ResNet(nn.Module):
    def __init__(self, args, model_name='resnet18'):
        super().__init__()
        self.device = args['device']
        self.model_name = model_name

        self.net = torch.hub.load('pytorch/vision:v0.10.0', self.model_name, pretrained=True)
        self.net.fc = nn.Linear(512, args['num_classes'])
        self.net.to(self.device)

    def forward(self, x):
        return self.net(x)

args = {
    'device': torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    'num_classes': 10,
    'learning_rate': 0.001,
    'num_epochs': 100,
    'batch_size': 32
}

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])
train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

model = ResNet(args)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=args['learning_rate'])

tasks = [
    [0, 1],  
    [2, 3],  
    [4, 5], 
    [6, 7],  
    [8, 9]   
]

def get_task_data(dataset, task_classes):
    indices = [i for i, (_, label) in enumerate(dataset) if label in task_classes]
    return Subset(dataset, indices)

def train_on_task(model, train_loader, criterion, optimizer, num_epochs):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(args['device']), labels.to(args['device'])
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}")

def evaluate_model(model, data_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(args['device']), labels.to(args['device'])
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            print(labels)
            print(predicted)
            print("##############")
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    accuracy = 100 * correct / total
    return accuracy

for i, task_classes in enumerate(tasks):
    print(f"\nTraining on Task {i} with classes: {task_classes}")
    
    task_train_data = get_task_data(train_dataset, task_classes)
    task_test_data = get_task_data(test_dataset, task_classes)
    train_loader = DataLoader(task_train_data, batch_size=args['batch_size'], shuffle=True)
    test_loader = DataLoader(task_test_data, batch_size=args['batch_size'], shuffle=False)
    
    train_on_task(model, train_loader, criterion, optimizer, args['num_epochs'])
    
    task_accuracy = evaluate_model(model, test_loader)
    print(f"Task {i} Accuracy after training: {task_accuracy:.2f}%")

print("\nEvaluating on the entire CIFAR-10 test dataset.")
full_test_loader = DataLoader(test_dataset, batch_size=args['batch_size'], shuffle=False)
overall_accuracy = evaluate_model(model, full_test_loader)
print(f"Overall Test Accuracy on CIFAR-10: {overall_accuracy:.2f}%")
