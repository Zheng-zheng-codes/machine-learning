import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset, TensorDataset

BATCH_SIZE = 256  # 每次给模型的图片个数
EPOCHS = 10       # 训练轮数
LR = 0.01         # 学习率
DEVICE = torch.device("cpu")

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

full_train = datasets.MNIST('data', train=True, download=True, transform=transform)
full_test  = datasets.MNIST('data', train=False, download=False, transform=transform)

# 筛选0/1标签
train_idx = (full_train.targets == 0) | (full_train.targets == 1)
test_idx  = (full_test.targets == 0) | (full_test.targets == 1)

train_data = Subset(full_train, torch.nonzero(train_idx).squeeze())
test_data  = Subset(full_test, torch.nonzero(test_idx).squeeze())

def apply_pca(data_list, k=50):
    """
    data_list: Subset or list of (img, label)
    k: 降维后维度
    返回: data_reduced, labels, components, mean
    """
    data = []
    labels = []
    for img, label in data_list:
        data.append(img.view(-1))
        labels.append(label)
    data = torch.stack(data)  # (N, 784)
    labels = torch.tensor(labels)

    mean = data.mean(dim=0)
    data_centered = data - mean

    # SVD 做 PCA
    U, S, V = torch.svd(data_centered)
    components = V[:, :k]  # (784, k)

    data_reduced = torch.mm(data_centered, components)  # (N, k)
    return data_reduced, labels, components, mean

k = 50  # 降维后的维度
train_X, train_y, components, mean = apply_pca(train_data, k)

# 测试集投影到训练集 PCA 空间
test_data_flat = []
test_labels = []
for img, label in test_data:
    test_data_flat.append(img.view(-1))
    test_labels.append(label)
test_data_flat = torch.stack(test_data_flat)           # (N_test, 784)
test_labels = torch.tensor(test_labels)

test_X = test_data_flat - mean
test_X = torch.mm(test_X, components)
test_y = test_labels

train_dataset = TensorDataset(train_X, train_y)
test_dataset  = TensorDataset(test_X, test_y)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

class LogisticRegression(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)
    def forward(self, x):
        return torch.sigmoid(self.linear(x))

class Perceptron(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)
    def forward(self, x):
        return self.linear(x)

class LinearSVM(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)
    def forward(self, x):
        return self.linear(x)

LogisticRegression_loss = nn.BCELoss()  # 交叉熵

def perceptron_loss(output, target):
    target = target.float()*2 - 1
    return torch.clamp(-output.squeeze()*target, min=0).mean()

def svm_loss(output, target):
    target = target.float()*2 - 1
    return torch.clamp(1 - output.squeeze()*target, min=0).mean()

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
    correct, total = 0, 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            output = model(data)
            pred = (output.squeeze() > 0.5).long()
            correct += (pred == target).sum().item()
            total += target.size(0)
    print(f"LOGISTIC REGRESSION Test Accuracy: {correct/total*100:.2f}%\n")

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
    correct, total = 0, 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            output = model(data)
            pred = (output.squeeze() > 0).long()
            correct += (pred == target).sum().item()
            total += target.size(0)
    print(f"PERCEPTRON Test Accuracy: {correct/total*100:.2f}%\n")

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
    correct, total = 0, 0
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
    lr_model = LogisticRegression(k).to(DEVICE)
    optimizer = optim.SGD(lr_model.parameters(), lr=LR, weight_decay=1e-4)
    print("Training Logistic Regression...")
    train_logistic_regression(lr_model, train_loader, LogisticRegression_loss, optimizer)
    test_logistic_regression(lr_model, test_loader)

    # Perceptron
    per_model = Perceptron(k).to(DEVICE)
    optimizer = optim.SGD(per_model.parameters(), lr=LR, weight_decay=1e-4)
    print("Training Perceptron...")
    train_perceptron(per_model, train_loader, optimizer)
    test_perceptron(per_model, test_loader)

    # Linear SVM
    svm_model = LinearSVM(k).to(DEVICE)
    optimizer = optim.SGD(svm_model.parameters(), lr=LR, weight_decay=1e-4)
    print("Training Linear SVM...")
    train_svm(svm_model, train_loader, optimizer)
    test_svm(svm_model, test_loader)