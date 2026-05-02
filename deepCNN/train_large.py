from dataset import get_loader

from model_large import SimpleResNet
from model_plain_large import SimpleCNN
from model_dropout_large import DropoutResNet

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt

def train(model, loader, optimizer, criterion, device):
    model.train()
    total_loss=0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)#放入设备
        optimizer.zero_grad()#清空梯度
        output = model(x)#向前传播
        loss = criterion(output,y)#计算损失
        loss.backward()#反向传播
        optimizer.step()#更新参数
        total_loss += loss.item()
    avg_loss = total_loss / len(loader)
    print(f"Loss = {avg_loss:.6f}")
    return avg_loss

def test(model, loader, device):
    model.eval()
    correct = 0
    total = 0#用来计算准确率
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)#放入设备
            output = model(x)
            pred = output.argmax(dim = 1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return correct/total

LR=0.001
epochs = 20
epochs2 = 50
epoch_list = list(range(1,epochs+1))
epoch2_list = list(range(1,epochs2+1))


loss_Simple10 = []
loss_CNN10 = []
loss_Dropout10 = []
result10 = {}

loss_Simple100 = []
loss_CNN100 = []
loss_Dropout100 = []
result100 = {}

train_data10 = get_loader("cifar10")
test_data10 = get_loader("cifar10", train = False)

train_data100 = get_loader("cifar100")
test_data100 = get_loader("cifar100", train = False)

def paint_loss():
    plt.figure()

    plt.plot(epoch_list,loss_Simple10,color='red',marker='o',label='SimpleResNet')
    plt.plot(epoch_list,loss_CNN10,color='blue',marker='s',label='SimpleCNN')
    plt.plot(epoch_list,loss_Dropout10,color='green',marker='^',label='DropoutResNet')

    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Comparison in CIFAR-10 for Large Model')

    plt.legend()
    plt.grid(True,linestyle='--',alpha=0.5)

    plt.savefig('loss_cifar10_large.png',dpi=300)
    plt.show()

    plt.figure()

    plt.plot(epoch2_list,loss_Simple100,color='red',marker='o',label='SimpleResNet')
    plt.plot(epoch2_list,loss_CNN100,color='blue',marker='s',label='SimpleCNN')
    plt.plot(epoch2_list,loss_Dropout100,color='green',marker='^',label='DropoutResNet')

    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Comparison in CIFAR-100 for Large Model')

    plt.legend()
    plt.grid(True,linestyle='--',alpha=0.5)

    plt.savefig('loss_cifar100_large.png',dpi=300)
    plt.show()

def paint_acc():
    plt.figure()

    names=list(result10.keys())
    acc=list(result10.values())

    plt.bar(names,acc,color=['red','blue','green'],edgecolor='black')
    
    plt.xlabel('Model Type')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Comparison in CIFAR-10 for Large Model')

    plt.ylim(0,1)
    plt.grid(axis='y',linestyle='--',alpha=0.5)
    for i,v in enumerate(acc):
        plt.text(i,v+0.0002,f"{v:.4f}",ha='center')

    plt.savefig('acc_cifar10_large.png',dpi=300)
    plt.show()

    plt.figure()

    names=list(result100.keys())
    acc=list(result100.values())

    plt.bar(names,acc,color=['red','blue','green'],edgecolor='black')
    
    plt.xlabel('Model Type')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Comparison in CIFAR-100 for Large Model')

    plt.ylim(0,1)
    plt.grid(axis='y',linestyle='--',alpha=0.5)
    for i,v in enumerate(acc):
        plt.text(i,v+0.0002,f"{v:.4f}",ha='center')

    plt.savefig('acc_cifar100_large.png',dpi=300)
    plt.show()

if __name__ == "__main__":
    device = "cpu"
    model1 = SimpleResNet().to(device)
    model2 = SimpleCNN().to(device)
    model3 = DropoutResNet().to(device)

    model4 = SimpleResNet(num_class = 100).to(device)
    model5 = SimpleCNN(num_class = 100).to(device)
    model6 = DropoutResNet(num_class = 100).to(device)

    optimizer1 = torch.optim.Adam(model1.parameters(), lr=LR)
    optimizer2 = torch.optim.Adam(model2.parameters(), lr=LR)
    optimizer3 = torch.optim.Adam(model3.parameters(), lr=LR)

    optimizer4 = torch.optim.Adam(model4.parameters(), lr=LR)
    optimizer5 = torch.optim.Adam(model5.parameters(), lr=LR)
    optimizer6 = torch.optim.Adam(model6.parameters(), lr=LR)

    scheduler1 = torch.optim.lr_scheduler.StepLR(optimizer1, step_size=15, gamma=0.5)
    scheduler2 = torch.optim.lr_scheduler.StepLR(optimizer2, step_size=15, gamma=0.5)
    scheduler3 = torch.optim.lr_scheduler.StepLR(optimizer3, step_size=15, gamma=0.5)

    scheduler4 = torch.optim.lr_scheduler.StepLR(optimizer4, step_size=15, gamma=0.5)
    scheduler5 = torch.optim.lr_scheduler.StepLR(optimizer5, step_size=15, gamma=0.5)
    scheduler6 = torch.optim.lr_scheduler.StepLR(optimizer6, step_size=15, gamma=0.5)

    criterion=nn.CrossEntropyLoss()
    for epoch in range(1,epochs+1):
        print(f"第{epoch}轮训练:") 
        loss1 = train(model1, train_data10, optimizer1, criterion, device)
        loss_Simple10.append(loss1)
        loss2 = train(model2, train_data10, optimizer2, criterion, device)
        loss_CNN10.append(loss2)
        loss3 = train(model3, train_data10, optimizer3, criterion, device)
        loss_Dropout10.append(loss3)

        scheduler1.step()
        scheduler2.step()
        scheduler3.step()
    
    acc1 = test(model1,test_data10,device)
    acc2 = test(model2,test_data10,device)
    acc3 = test(model3,test_data10,device)
    result10["SimpleResNet"] = acc1
    result10["SimpleCNN"] = acc2
    result10["DropoutResNet"] = acc3

    for epoch in range(1,epochs2+1):
        print(f"第{epoch}轮训练:") 
        loss4 = train(model4, train_data100, optimizer4, criterion, device)
        loss_Simple100.append(loss4)
        loss5 = train(model5, train_data100, optimizer5, criterion, device)
        loss_CNN100.append(loss5)
        loss6 = train(model6, train_data100, optimizer6, criterion, device)
        loss_Dropout100.append(loss6)

        scheduler4.step()
        scheduler5.step()
        scheduler6.step()

    acc4 = test(model4,test_data100,device)
    acc5 = test(model5,test_data100,device)
    acc6 = test(model6,test_data100,device)
    result100["SimpleResNet"] = acc4
    result100["SimpleCNN"] = acc5
    result100["DropoutResNet"] = acc6

    paint_loss()
    paint_acc()