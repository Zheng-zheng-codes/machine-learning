#这个代码是基于老师的设计没有改动
import torch
import torch.nn as nn
from torchvision import datasets,transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

#数据预处理
train_transform=transforms.Compose([
    transforms.RandomAffine(
        degrees=10,
        translate=(0.1,0.1),
        scale=(0.9,1.1),
        shear=10
        ),#随机仿射变换
    #degrees：随机旋转-10°~10°
    #translate：像素平移一点
    #scale：放大/缩小10%
    #shear：剪切变形10°
    transforms.ToTensor(),#转到0~1
    transforms.Normalize(mean=[0.1307],std=[0.3081])#标准化
])
test_transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.1307],std=[0.3081])
])

#读取训练集和测试集
train_data=datasets.MNIST(
    root="data",
    train=True,
    download=True,
    transform=train_transform
)
test_data=datasets.MNIST(
    root="data",
    train=False,
    download=True,
    transform=test_transform
)

#加载测试和训练数据
train_loader=DataLoader(
    train_data,
    batch_size=64,
    shuffle=True
)
test_loader=DataLoader(
    test_data,
    batch_size=1000,
    shuffle=False
)

LR=0.001#学习率
epochs=10#训练轮数
epoch_list=list(range(1,epochs+1))

#模型构建
class CNN(nn.Module):
    def __init__(self,act):
        super().__init__()
        self.act=act
        self.conv1=nn.Conv2d(1,6,5,padding=2)
        #在图片外围补两圈（保持28*28不变），用六种5*5的卷积核扫描得到六种28*28的特征表示
        self.pool1=nn.MaxPool2d(2,2)#降采样28*28->14*14
        self.conv2=nn.Conv2d(6,16,5)
        #图变成了10*10，6种特征组合成了16种特征
        self.pool2=nn.MaxPool2d(2,2)#降采样10*10->5*5
        self.conv3=nn.Conv2d(16,120,5)
        #图变成了1*1，16种特征组成了120种特征
        self.fc1=nn.Linear(120,84)
        self.fc2=nn.Linear(84,10)
    def forward(self,x):
        x=self.conv1(x)
        x=self.act(x)
        x=self.pool1(x)
        x=self.conv2(x)
        x=self.act(x)
        x=self.pool2(x)
        x=self.conv3(x)
        x=x.view(x.size(0),-1)
        x=self.fc1(x)
        x=self.act(x)
        x=self.fc2(x)
        return x

#各种激活函数
acts={
    "ReLU":nn.ReLU(),
    "GELU":nn.GELU(),
    "Tanh":nn.Tanh(),
    "ELU":nn.ELU(),
}
loss_ReLU=[]
loss_GELU=[]
loss_Tanh=[]
loss_ELU=[]
result={}

#训练函数
def train(model,loader,optimizer,criterion,device):
    model.train()
    total_loss=0
    for x,y in loader:
        x=x.to(device)
        y=y.to(device)#放入设备
        optimizer.zero_grad()#清空梯度
        output=model(x)#向前传播
        loss=criterion(output,y)#计算损失
        loss.backward()#反向传播
        optimizer.step()#更新参数
        total_loss+=loss.item()
    avg_loss=total_loss/len(loader)
    print(f"Loss={avg_loss:.6f}")
    return avg_loss

#测试函数
def test(model,loader,device):
    model.eval()
    correct=0
    total=0#用来计算准确率
    with torch.no_grad():
        for x,y in loader:
            x=x.to(device)
            y=y.to(device)#放入设备
            output=model(x)
            pred=output.argmax(dim=1)
            correct+=(pred==y).sum().item()
            total+=y.size(0)
    return correct/total

#绘图
def print_loss():
    plt.figure()

    plt.plot(epoch_list,loss_ReLU,color='red',marker='o',label='ReLU')
    plt.plot(epoch_list,loss_GELU,color='blue',marker='s',label='GELU')
    plt.plot(epoch_list,loss_Tanh,color='green',marker='^',label='Tanh')
    plt.plot(epoch_list,loss_ELU,color='orange',marker='d',label='ELU')

    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Comparison')

    plt.legend()
    plt.grid(True,linestyle='--',alpha=0.5)

    plt.savefig('loss_origin.png',dpi=300)
    plt.show()

def print_acc():
    plt.figure()

    names=list(result.keys())
    acc=list(result.values())

    plt.bar(names,acc,color=['red','blue','green','orange'],edgecolor='black')
    
    plt.xlabel('Activation Function')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Comparison')

    plt.ylim(0.98,1.0)
    plt.grid(axis='y',linestyle='--',alpha=0.5)
    for i,v in enumerate(acc):
        plt.text(i,v+0.0002,f"{v:.4f}",ha='center')

    plt.savefig('acc_origin.png',dpi=300)
    plt.show()

if __name__=="__main__":
    device="cpu"
    for name,act in acts.items():#遍历每一种激活函数
        model=CNN(act).to(device)#创建模型
        optimizer=torch.optim.Adam(model.parameters(), lr=LR)#优化器
        criterion=nn.CrossEntropyLoss()#损失函数
        print(f"激活函数：{name}")
        for epoch in range(1,epochs+1):
            print(f"第{epoch}轮训练:")
            loss=train(model,train_loader,optimizer,criterion,device)
            if name=="ReLU":
                loss_ReLU.append(loss)
            elif name=="GELU":
                loss_GELU.append(loss)
            elif name=="Tanh":
                loss_Tanh.append(loss)
            elif name=="ELU":
                loss_ELU.append(loss)
        acc=test(model,test_loader,device)
        if name=="ReLU":
            result["ReLU"]=acc
        elif name=="GELU":
            result["GELU"]=acc
        elif name=="Tanh":
            result["Tanh"]=acc
        elif name=="ELU":
            result["ELU"]=acc
        print(f"{name}准确率：{acc}\n")
    print_loss()
    print_acc()