import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

BATCH_SIZE = 256 #每次给模型的图片个数
EPOCHS = 10 #一共进行的轮数
LR = 0.01 #学习率
DEVICE = torch.device("cpu")

#数据加载
transform = transforms.Compose([
    transforms.ToTensor(),#归一化
    transforms.Normalize((0.1307,), (0.3081,))#标准化
])

full_train = datasets.MNIST('data', train=True, download=False, transform=transform)#读取训练集
full_test  = datasets.MNIST('data', train=False, download=False, transform=transform)#读取测试集

#筛选0/1标签
train_idx = (full_train.targets == 0) | (full_train.targets == 1)
test_idx  = (full_test.targets == 0) | (full_test.targets == 1)

train_data = Subset(full_train, torch.nonzero(train_idx).squeeze())
test_data  = Subset(full_test, torch.nonzero(test_idx).squeeze())
#nonzero:筛选true的样例
#squeeze：去掉多余的维度

train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
test_loader  = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)

#逻辑回归
class LogisticRegression(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(28*28, 1)
    def forward(self, x):
        x = x.view(x.size(0), -1)
        return torch.sigmoid(self.linear(x))

#感知机
class Perceptron(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(28*28, 1)
    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.linear(x)

#支持向量机
class LinearSVM(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(28*28, 1)
    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.linear(x)

#损失函数

#逻辑回归
LogisticRegression_loss = nn.BCELoss()#交叉熵

#感知机
def perceptron_loss(output, target):
    target = target.float()*2 - 1  # 0/1 -> -1/1
    return torch.clamp(-output.squeeze()*target, min=0).mean()

#支持向量机
def svm_loss(output, target):
    target = target.float()*2 - 1
    return torch.clamp(1 - output.squeeze()*target, min=0).mean()

#训练函数和测试函数

# Logistic Regression
def train_logistic_regression(model, train_loader, criterion, optimizer, epochs=EPOCHS):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for data, target in train_loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            target = target.view(-1, 1).float()  
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.4f}")

def test_logistic_regression(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            output = model(data)
            pred = (output.squeeze() > 0.5).long()
            correct += (pred == target).sum().item()
            total += target.size(0)
    print(f"LOGISTIC REGRESSION Test Accuracy: {correct/total*100:.2f}%\n")

# Perceptron
def train_perceptron(model, train_loader, optimizer, epochs=EPOCHS):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for data, target in train_loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            optimizer.zero_grad()
            output = model(data)
            loss = perceptron_loss(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.4f}")

def test_perceptron(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            output = model(data)
            pred = (output.squeeze() > 0).long()
            correct += (pred == target).sum().item()
            total += target.size(0)
    print(f"PERCEPTRON Test Accuracy: {correct/total*100:.2f}%\n")

# Linear SVM
def train_svm(model, train_loader, optimizer, epochs=EPOCHS):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for data, target in train_loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            optimizer.zero_grad()
            output = model(data)
            loss = svm_loss(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.4f}")

def test_svm(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            output = model(data)
            pred = (output.squeeze() > 0).long()
            correct += (pred == target).sum().item()
            total += target.size(0)
    print(f"LINEAR SVM Test Accuracy: {correct/total*100:.2f}%\n")

if __name__ == "__main__":
    # Logistic Regression
    lr_model = LogisticRegression().to(DEVICE)
    optimizer = optim.SGD(lr_model.parameters(), lr=LR, weight_decay=1e-4)
    print("Training Logistic Regression...")
    train_logistic_regression(lr_model, train_loader, LogisticRegression_loss, optimizer)
    test_logistic_regression(lr_model, test_loader)

    # Perceptron
    per_model = Perceptron().to(DEVICE)
    optimizer = optim.SGD(per_model.parameters(), lr=LR, weight_decay=1e-4)
    print("Training Perceptron...")
    train_perceptron(per_model, train_loader, optimizer)
    test_perceptron(per_model, test_loader)

    # Linear SVM
    svm_model = LinearSVM().to(DEVICE)
    optimizer = optim.SGD(svm_model.parameters(), lr=LR, weight_decay=1e-4)
    print("Training Linear SVM...")
    train_svm(svm_model, train_loader, optimizer)
    test_svm(svm_model, test_loader)