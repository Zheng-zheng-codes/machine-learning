from dataset import get_loader

from model import DropoutResNet

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import copy

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
    ans = correct / total
    print(f"Accuracy = {ans:.4f}")
    return ans

def get_optimizer(name, model):

    if name == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=0.01
        )

    elif name == "momentum":
        return torch.optim.SGD(
            model.parameters(),
            lr=0.01,
            momentum=0.9
        )

    elif name == "adam":
        return torch.optim.Adam(
            model.parameters(),
            lr=0.001
        )

def save_model(model, name):
    torch.save(model.state_dict(), f"{name}.pth")

epochs10 = 10
epoch_list10 = list(range(1,epochs10+1))

epochs100 = 20
epoch_list100 = list(range(1,epochs100+1))

epochs200 = 30
epoch_list200 = list(range(1,epochs200+1))

loss_SGD10 = []
loss_Momentum10 = []
loss_Adam10 = []
result10 = {}

loss_SGD100 = []
loss_Momentum100 = []
loss_Adam100 = []
result100 = {}

loss_SGD200 = [] 
loss_Momentum200 = [] 
loss_Adam200 = [] 
result200 = {}

train_data10 = get_loader("cifar10")
test_data10 = get_loader("cifar10", train = False)

train_data100 = get_loader("cifar100")
test_data100 = get_loader("cifar100", train = False)

train_data200 = get_loader("tinyimagenet")
test_data200 = get_loader("tinyimagenet", train = False)

def paint_loss():
    plt.figure()

    plt.plot(epoch_list10,loss_SGD10,color='red',marker='o',label='SGD')
    plt.plot(epoch_list10,loss_Momentum10,color='blue',marker='s',label='Momentum')
    plt.plot(epoch_list10,loss_Adam10,color='green',marker='^',label='Adam')

    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Comparison in CIFAR-10')

    plt.legend()
    plt.grid(True,linestyle='--',alpha=0.5)

    plt.savefig('loss_cifar10.png',dpi=300)
    plt.show()

    plt.figure()

    plt.plot(epoch_list100,loss_SGD100,color='red',marker='o',label='SGD')
    plt.plot(epoch_list100,loss_Momentum100,color='blue',marker='s',label='Momentum')
    plt.plot(epoch_list100,loss_Adam100,color='green',marker='^',label='Adam')

    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Comparison in CIFAR-100')

    plt.legend()
    plt.grid(True,linestyle='--',alpha=0.5)

    plt.savefig('loss_cifar100.png',dpi=300)
    plt.show()

    plt.figure()
    plt.plot(epoch_list200,loss_SGD200,color='red',marker='o',label='SGD')
    plt.plot(epoch_list200,loss_Momentum200,color='blue',marker='s',label='Momentum')
    plt.plot(epoch_list200,loss_Adam200,color='green',marker='^',label='Adam')

    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Comparison in CIFAR-200')

    plt.legend()
    plt.grid(True,linestyle='--',alpha=0.5)

    plt.savefig('loss_cifar200.png',dpi=300)
    plt.show()

def paint_acc():
    plt.figure()

    names=list(result10.keys())
    acc=list(result10.values())

    plt.bar(names,acc,color=['red','blue','green'],edgecolor='black')
    
    plt.xlabel('Model Type')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Comparison in CIFAR-10')

    plt.ylim(0,1)
    plt.grid(axis='y',linestyle='--',alpha=0.5)
    for i,v in enumerate(acc):
        plt.text(i,v+0.0002,f"{v:.4f}",ha='center')

    plt.savefig('acc_cifar10.png',dpi=300)
    plt.show()

    plt.figure()

    names=list(result100.keys())
    acc=list(result100.values())

    plt.bar(names,acc,color=['red','blue','green'],edgecolor='black')
    
    plt.xlabel('Model Type')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Comparison in CIFAR-100')

    plt.ylim(0,1)
    plt.grid(axis='y',linestyle='--',alpha=0.5)
    for i,v in enumerate(acc):
        plt.text(i,v+0.0002,f"{v:.4f}",ha='center')

    plt.savefig('acc_cifar100.png',dpi=300)
    plt.show()

    plt.figure()

    names=list(result200.keys())
    acc=list(result200.values())

    plt.bar(names,acc,color=['red','blue','green'],edgecolor='black')

    plt.xlabel('Model Type')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Comparison in CIFAR-200')

    plt.ylim(0,1)
    plt.grid(axis='y',linestyle='--',alpha=0.5)
    for i,v in enumerate(acc):
        plt.text(i,v+0.0002,f"{v:.4f}",ha='center')

    plt.savefig('acc_cifar200.png',dpi=300)
    plt.show()

if __name__ == "__main__":
    device = "cpu"

    base_model1 = DropoutResNet().to(device)
    base_model2 = DropoutResNet(num_class = 100).to(device)
    base_model3 = DropoutResNet(num_class = 200).to(device)

    model1 = copy.deepcopy(base_model1)
    model2 = copy.deepcopy(base_model1)
    model3 = copy.deepcopy(base_model1)

    model4 = copy.deepcopy(base_model2)
    model5 = copy.deepcopy(base_model2)
    model6 = copy.deepcopy(base_model2)

    model7 = copy.deepcopy(base_model3)
    model8 = copy.deepcopy(base_model3)
    model9 = copy.deepcopy(base_model3)

    optimizer1 = get_optimizer("sgd", model1)
    optimizer2 = get_optimizer("momentum", model2)
    optimizer3 = get_optimizer("adam", model3)

    optimizer4 = get_optimizer("sgd", model4)
    optimizer5 = get_optimizer("momentum", model5)
    optimizer6 = get_optimizer("adam", model6)

    optimizer7 = get_optimizer("sgd", model7)
    optimizer8 = get_optimizer("momentum", model8)
    optimizer9 = get_optimizer("adam", model9)
    
    criterion=nn.CrossEntropyLoss()
    for epoch in range(1,epochs10+1):
        print(f"第{epoch}轮训练:") 
        loss1 = train(model1, train_data10, optimizer1, criterion, device)
        loss_SGD10.append(loss1)
        loss2 = train(model2, train_data10, optimizer2, criterion, device)
        loss_Momentum10.append(loss2)
        loss3 = train(model3, train_data10, optimizer3, criterion, device)
        loss_Adam10.append(loss3)
    
    acc1 = test(model1,test_data10,device)
    torch.save(model1.state_dict(), "cifar10_sgd.pth")
    acc2 = test(model2,test_data10,device)
    torch.save(model2.state_dict(), "cifar10_momentum.pth")
    acc3 = test(model3,test_data10,device)
    torch.save(model3.state_dict(), "cifar10_adam.pth")
    result10["SGD"] = acc1
    result10["Momentum"] = acc2
    result10["Adam"] = acc3

    for epoch in range(1,epochs100+1):
        print(f"第{epoch}轮训练:") 
        loss4 = train(model4, train_data100, optimizer4, criterion, device)
        loss_SGD100.append(loss4)
        loss5 = train(model5, train_data100, optimizer5, criterion, device)
        loss_Momentum100.append(loss5)
        loss6 = train(model6, train_data100, optimizer6, criterion, device)
        loss_Adam100.append(loss6)

    acc4 = test(model4,test_data100,device)
    torch.save(model4.state_dict(), "cifar100_sgd.pth")
    acc5 = test(model5,test_data100,device)
    torch.save(model5.state_dict(), "cifar100_momentum.pth")
    acc6 = test(model6,test_data100,device)
    torch.save(model6.state_dict(), "cifar100_adam.pth")
    result100["SGD"] = acc4
    result100["Momentum"] = acc5
    result100["Adam"] = acc6

    for epoch in range(1,epochs200+1): 
        print(f"第{epoch}轮训练:") 
        loss7 = train(model7, train_data200, optimizer7, criterion, device)
        loss_SGD200.append(loss7)
        loss8 = train(model8, train_data200, optimizer8, criterion, device)
        loss_Momentum200.append(loss8)
        loss9 = train(model9, train_data200, optimizer9, criterion, device)
        loss_Adam200.append(loss9)

    acc7 = test(model7,test_data200,device)
    torch.save(model7.state_dict(), "tinyimagenet_sgd.pth")
    acc8 = test(model8,test_data200,device)
    torch.save(model8.state_dict(), "tinyimagenet_momentum.pth")
    acc9 = test(model9,test_data200,device)
    torch.save(model9.state_dict(), "tinyimagenet_adam.pth")
    result200["SGD"] = acc7
    result200["Momentum"] = acc8
    result200["Adam"] = acc9

    paint_loss()
    paint_acc()