from dataset import get_loader
from model import ResNet18

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import csv


# 单轮训练
def train(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(loader)
    return avg_loss


# 测试准确率
def test(model, loader, device):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            output = model(x)
            pred = output.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)

    acc = correct / total
    return acc


# 实验要求：以 Momentum SGD 为例
def get_optimizer(model, lr=0.1):
    return torch.optim.SGD(
        model.parameters(),
        lr=lr,
        momentum=0.9,
        weight_decay=5e-4
    )


# 三种学习率策略：阶梯衰减、线性衰减、Warm up
def get_scheduler(name, optimizer, epochs, warmup_epochs=5):
    if name == "step":
        # 每到指定轮数，学习率乘以 0.1
        return torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=[epochs // 2, int(epochs * 0.75)],
            gamma=0.1
        )

    elif name == "linear":
        # 从初始学习率线性衰减到 0
        return torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda epoch: 1 - epoch / epochs
        )

    elif name == "warmup":
        # 前 warmup_epochs 轮逐渐升高学习率，之后使用余弦衰减
        def lr_lambda(epoch):
            if epoch < warmup_epochs:
                return (epoch + 1) / warmup_epochs
            else:
                return 0.5 * (1 + torch.cos(torch.tensor((epoch - warmup_epochs) / (epochs - warmup_epochs) * 3.1415926))).item()

        return torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lr_lambda
        )

    else:
        raise ValueError("scheduler 名称只能是 step、linear 或 warmup")


# 训练一个实验配置
def run_experiment(exp_name, dataset_name, num_classes, norm_type, scheduler_name, epochs, batch_size, device):
    print(f"\n========== 开始实验：{exp_name} ==========")

    train_loader = get_loader(dataset_name, train=True, batch_size=batch_size)
    test_loader = get_loader(dataset_name, train=False, batch_size=batch_size)

    model = ResNet18(num_classes=num_classes, norm_type=norm_type).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = get_optimizer(model, lr=0.1)
    scheduler = get_scheduler(scheduler_name, optimizer, epochs)

    loss_list = []
    acc_list = []
    lr_list = []

    for epoch in range(1, epochs + 1):
        loss = train(model, train_loader, optimizer, criterion, device)
        acc = test(model, test_loader, device)
        current_lr = optimizer.param_groups[0]["lr"]

        loss_list.append(loss)
        acc_list.append(acc)
        lr_list.append(current_lr)

        print(
            f"第 {epoch:02d}/{epochs} 轮 | "
            f"Loss = {loss:.6f} | "
            f"Accuracy = {acc:.4f} | "
            f"LR = {current_lr:.6f}"
        )

        scheduler.step()

    torch.save(model.state_dict(), f"{exp_name}.pth")
    save_result_csv(exp_name, loss_list, acc_list, lr_list)

    return loss_list, acc_list, lr_list


# 保存结果，方便写实验报告
def save_result_csv(exp_name, loss_list, acc_list, lr_list):
    with open(f"{exp_name}.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "loss", "accuracy", "learning_rate"])

        for i in range(len(loss_list)):
            writer.writerow([i + 1, loss_list[i], acc_list[i], lr_list[i]])


# 画学习率曲线
def paint_lr(result_dict, filename, title):
    plt.figure()

    for name, values in result_dict.items():
        plt.plot(range(1, len(values) + 1), values, marker="o", label=name)

    plt.xlabel("Epoch")
    plt.ylabel("Learning Rate")
    plt.title(title)
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.savefig(filename, dpi=300)
    plt.show()


# 画 loss 曲线
def paint_loss(result_dict, filename, title):
    plt.figure()

    for name, values in result_dict.items():
        plt.plot(range(1, len(values) + 1), values, marker="o", label=name)

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(title)
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.savefig(filename, dpi=300)
    plt.show()


# 画 accuracy 曲线
def paint_acc(result_dict, filename, title):
    plt.figure()

    for name, values in result_dict.items():
        plt.plot(range(1, len(values) + 1), values, marker="o", label=name)

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title(title)
    plt.ylim(0, 1)
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.savefig(filename, dpi=300)
    plt.show()


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("当前设备：", device)

    epochs = 20
    batch_size = 128

    dataset_name = "cifar10"
    num_classes = 10

    # 实验一：学习率策略对比
    # 固定模型为 BN-ResNet18，固定优化器为 Momentum SGD，只改变学习率策略
    lr_loss_result = {}
    lr_acc_result = {}
    lr_curve_result = {}

    for scheduler_name in ["step", "linear", "warmup"]:
        exp_name = f"cifar10_bn_{scheduler_name}"

        loss_list, acc_list, lr_list = run_experiment(
            exp_name=exp_name,
            dataset_name=dataset_name,
            num_classes=num_classes,
            norm_type="bn",
            scheduler_name=scheduler_name,
            epochs=epochs,
            batch_size=batch_size,
            device=device
        )

        lr_loss_result[scheduler_name] = loss_list
        lr_acc_result[scheduler_name] = acc_list
        lr_curve_result[scheduler_name] = lr_list

    paint_lr(
        lr_curve_result,
        "lr_strategy_compare.png",
        "Learning Rate Schedule Comparison"
    )

    paint_loss(
        lr_loss_result,
        "loss_lr_compare.png",
        "Loss Comparison of Learning Rate Strategies"
    )

    paint_acc(
        lr_acc_result,
        "acc_lr_compare.png",
        "Accuracy Comparison of Learning Rate Strategies"
    )

    # 实验二：BN 和 LN 对比
    # 固定优化器为 Momentum SGD，固定学习率策略为 StepLR，只改变归一化方式
    norm_loss_result = {}
    norm_acc_result = {}

    for norm_type in ["bn", "ln"]:
        exp_name = f"cifar10_{norm_type}_step"

        loss_list, acc_list, lr_list = run_experiment(
            exp_name=exp_name,
            dataset_name=dataset_name,
            num_classes=num_classes,
            norm_type=norm_type,
            scheduler_name="step",
            epochs=epochs,
            batch_size=batch_size,
            device=device
        )

        norm_loss_result[norm_type.upper()] = loss_list
        norm_acc_result[norm_type.upper()] = acc_list

    paint_loss(
        norm_loss_result,
        "loss_norm_compare.png",
        "Loss Comparison of BN and LN"
    )

    paint_acc(
        norm_acc_result,
        "acc_norm_compare.png",
        "Accuracy Comparison of BN and LN"
    )
