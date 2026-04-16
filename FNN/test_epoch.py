import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from sklearn.decomposition import PCA
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt

#把原始数据转化为神经网络可以读的形式（PyTorch Tensor）并进行归一化
#加通道：28*28-->1*28*28
transform=transforms.ToTensor()

#读取训练集和测试集
train_data=datasets.MNIST(
    root="data",
    train=True,
    download=True,
    transform=transform
)
test_data=datasets.MNIST(
    root="data",
    train=False,
    download=True,
    transform=transform
)

#numpy:把数据转为NumPy数组格式便于后序矩阵/向量计算
#reshape:把每张图的数据28*28展开成784（-1表示自动计算）
#除以255.0:归一化
x_train=train_data.data.numpy().reshape(-1,784)/255.0
#提取每个数据的标签（0~9）
y_train=train_data.targets.numpy()

x_test=test_data.data.numpy().reshape(-1,784)/255.0
y_test=test_data.targets.numpy()

#降维
k=100
pca=PCA(n_components=k)
x_train=pca.fit_transform(x_train)
x_test=pca.transform(x_test)

#把数据转成tensor
x_train=torch.tensor(x_train,dtype=torch.float32)
y_train=torch.tensor(y_train,dtype=torch.long)

x_test=torch.tensor(x_test,dtype=torch.float32)
y_test=torch.tensor(y_test,dtype=torch.long)

train_dataset=TensorDataset(x_train, y_train)
train_loader=DataLoader(train_dataset, batch_size=128, shuffle=True)

#模型搭建
class FNN(nn.Module):
    def __init__(self,in_dim=k,out_dim=10,act='relu'):
        super(FNN, self).__init__()
        self.layer1=nn.Linear(in_dim,64)
        if act=='relu':
            self.act=nn.ReLU(inplace=True)
        elif act=='tanh':
            self.act=nn.Tanh()
        self.layer2=nn.Linear(64,out_dim)
        # self.softmax=nn.Softmax(dim=0)
        for params in self.parameters():
            nn.init.normal_(params, mean=0, std=0.01)
    def forward(self,x):
        x=self.layer1(x)
        x=self.act(x)
        x=self.layer2(x)
        # x=self.softmax(x)
        return x
model_relu=FNN(act='relu')
model_tanh=FNN(act='tanh')

#训练模型
LR=0.001
def train_model(model,epoch):
    loss=nn.CrossEntropyLoss()#损失函数
    optimizer=optim.Adam(model.parameters(), lr=LR)#优化器
    for i in range(epoch):
        for x_batch,y_batch in train_loader:
            outputs=model(x_batch)
            l=loss(outputs, y_batch)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
        # print(f"Epoch {epoch+1}, Loss: {l.item()}")
    return model

#测试模型
def test_model(model):
    model.eval()
    with torch.no_grad():
        outputs=model(x_test)
        _,pred=torch.max(outputs,1)
        acc=(pred==y_test).float().mean().item()
    return acc

relu_acc_list=[]
tanh_acc_list=[]
epochs=[5,10,15,20,25,30]
for epoch in epochs:
    model_relu=FNN(act='relu')
    model_tanh=FNN(act='tanh')
    print(f"{epoch} :\n")
    print("Training ReLU model...")
    model_relu=train_model(model_relu,epoch)

    print("Training Tanh model...")
    model_tanh=train_model(model_tanh,epoch)
    relu_acc_list.append(test_model(model_relu))
    tanh_acc_list.append(test_model(model_tanh))

plt.figure()

plt.plot(epochs, relu_acc_list, marker='o', label='ReLU')
plt.plot(epochs, tanh_acc_list, marker='s', label='Tanh')

plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Epochs vs Accuracy (PCA=100)')
plt.legend()
plt.grid(True)

plt.savefig("epoch_result.png")
plt.show()