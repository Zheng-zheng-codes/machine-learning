import os
import json
import time
import random
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim

import matplotlib.pyplot as plt

from dataset import get_dataloaders
from model import LSTMClassifier, count_parameters


# =========================
# 0. 固定随机种子
# =========================
def set_seed(seed=42):
    """
    固定随机种子，使实验结果尽量可复现。
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


# =========================
# 1. 单轮训练函数
# =========================
def train_one_epoch(model, train_loader, criterion, optimizer, device):
    """
    训练模型一个 epoch。

    返回：
    avg_loss: 当前 epoch 平均训练损失
    train_acc: 当前 epoch 训练准确率
    """

    model.train()

    total_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (x, lengths, y) in enumerate(train_loader):
        x = x.to(device)
        lengths = lengths.to(device)
        y = y.to(device)

        optimizer.zero_grad()

        outputs = model(x, lengths)

        loss = criterion(outputs, y)

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

        preds = torch.argmax(outputs, dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)

        # 每 50 个 batch 输出一次进度，避免长时间没反馈
        if (batch_idx + 1) % 50 == 0:
            print(f"已训练 batch: {batch_idx + 1}/{len(train_loader)}")

    avg_loss = total_loss / len(train_loader)
    train_acc = correct / total

    return avg_loss, train_acc


# =========================
# 2. 测试函数
# =========================
def evaluate(model, test_loader, criterion, device):
    """
    在测试集上评估模型。

    返回：
    avg_loss: 测试集平均损失
    test_acc: 测试集准确率
    """

    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for x, lengths, y in test_loader:
            x = x.to(device)
            lengths = lengths.to(device)
            y = y.to(device)

            outputs = model(x, lengths)

            loss = criterion(outputs, y)

            total_loss += loss.item()

            preds = torch.argmax(outputs, dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)

    avg_loss = total_loss / len(test_loader)
    test_acc = correct / total

    return avg_loss, test_acc


# =========================
# 3. 完整训练一个模型
# =========================
def run_experiment(
    model_name,
    bidirectional,
    train_loader,
    test_loader,
    vocab_size,
    device,
    epochs=5,
    embed_dim=128,
    hidden_dim=256,
    num_layers=1,
    learning_rate=1e-3
):
    """
    训练并测试一个模型。

    参数：
    model_name: 模型名称，例如 LSTM / Bi-LSTM
    bidirectional: 是否为双向 LSTM
    """

    print("\n" + "=" * 70)
    print(f"开始训练模型：{model_name}")
    print("=" * 70)

    model = LSTMClassifier(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_classes=2,
        bidirectional=bidirectional,
        dropout=0.5,
        pad_idx=0
    ).to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(
        model.parameters(),
        lr=learning_rate
    )

    params = count_parameters(model)

    print(f"模型名称：{model_name}")
    print(f"模型参数量：{params}")

    epoch_times = []
    train_losses = []
    train_accs = []
    test_losses = []
    test_accs = []

    for epoch in range(1, epochs + 1):
        start_time = time.time()

        train_loss, train_acc = train_one_epoch(
            model=model,
            train_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device
        )

        end_time = time.time()
        epoch_time = end_time - start_time

        test_loss, test_acc = evaluate(
            model=model,
            test_loader=test_loader,
            criterion=criterion,
            device=device
        )

        epoch_times.append(epoch_time)
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        test_losses.append(test_loss)
        test_accs.append(test_acc)

        print(
            f"Epoch [{epoch}/{epochs}] "
            f"Train Loss: {train_loss:.4f} "
            f"Train Acc: {train_acc * 100:.2f}% "
            f"Test Loss: {test_loss:.4f} "
            f"Test Acc: {test_acc * 100:.2f}% "
            f"Time: {epoch_time:.2f}s"
        )

    avg_epoch_time = sum(epoch_times) / len(epoch_times)
    final_test_acc = test_accs[-1]

    # =========================
    # 保存训练好的模型
    # =========================
    os.makedirs("checkpoints", exist_ok=True)

    save_path = f"checkpoints/{model_name}.pth"

    torch.save(
        {
            "model_name": model_name,
            "model_state_dict": model.state_dict(),

            # 模型结构参数
            "vocab_size": vocab_size,
            "embed_dim": embed_dim,
            "hidden_dim": hidden_dim,
            "num_layers": num_layers,
            "num_classes": 2,
            "bidirectional": bidirectional,

            # 实验结果
            "params": params,
            "final_test_acc": final_test_acc,
            "avg_epoch_time": avg_epoch_time,
            "epoch_times": epoch_times,
            "train_losses": train_losses,
            "train_accs": train_accs,
            "test_losses": test_losses,
            "test_accs": test_accs
        },
        save_path
    )

    print(f"模型已保存到: {save_path}")

    result = {
        "model_name": model_name,
        "params": params,
        "final_test_acc": final_test_acc,
        "avg_epoch_time": avg_epoch_time,
        "epoch_times": epoch_times,
        "train_losses": train_losses,
        "train_accs": train_accs,
        "test_losses": test_losses,
        "test_accs": test_accs,
        "save_path": save_path
    }

    return result


# =========================
# 4. 绘图函数
# =========================
def plot_results(lstm_result, bilstm_result, epochs):
    """
    绘制 LSTM 与 Bi-LSTM 的实验结果对比图。
    """

    epoch_list = list(range(1, epochs + 1))

    os.makedirs("figures", exist_ok=True)

    # =========================
    # 图 1：训练损失曲线
    # =========================
    plt.figure(figsize=(8, 5))

    plt.plot(
        epoch_list,
        lstm_result["train_losses"],
        marker="o",
        label="LSTM"
    )

    plt.plot(
        epoch_list,
        bilstm_result["train_losses"],
        marker="s",
        label="Bi-LSTM"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Training Loss")
    plt.title("Training Loss Comparison")
    plt.legend()
    plt.grid(True)

    plt.savefig(
        "figures/training_loss_comparison.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()

    # =========================
    # 图 2：测试准确率曲线
    # =========================
    plt.figure(figsize=(8, 5))

    plt.plot(
        epoch_list,
        [acc * 100 for acc in lstm_result["test_accs"]],
        marker="o",
        label="LSTM"
    )

    plt.plot(
        epoch_list,
        [acc * 100 for acc in bilstm_result["test_accs"]],
        marker="s",
        label="Bi-LSTM"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Test Accuracy (%)")
    plt.title("Test Accuracy Comparison")
    plt.legend()
    plt.grid(True)

    plt.savefig(
        "figures/test_accuracy_comparison.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()

    # =========================
    # 图 3：每轮训练时间柱状图
    # =========================
    x = np.arange(epochs)
    width = 0.35

    plt.figure(figsize=(8, 5))

    plt.bar(
        x - width / 2,
        lstm_result["epoch_times"],
        width=width,
        label="LSTM"
    )

    plt.bar(
        x + width / 2,
        bilstm_result["epoch_times"],
        width=width,
        label="Bi-LSTM"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Training Time (s)")
    plt.title("Training Time Comparison")
    plt.xticks(x, epoch_list)
    plt.legend()
    plt.grid(True, axis="y")

    plt.savefig(
        "figures/training_time_comparison.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()

    # =========================
    # 图 4：模型参数量柱状图
    # =========================
    model_names = ["LSTM", "Bi-LSTM"]

    params = [
        lstm_result["params"],
        bilstm_result["params"]
    ]

    plt.figure(figsize=(7, 5))

    bars = plt.bar(
        model_names,
        params
    )

    plt.xlabel("Model")
    plt.ylabel("Number of Parameters")
    plt.title("Model Parameters Comparison")
    plt.grid(True, axis="y")

    for bar in bars:
        height = bar.get_height()

        plt.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            str(height),
            ha="center",
            va="bottom"
        )

    plt.savefig(
        "figures/model_parameters_comparison.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()

    print("\n可视化图片已保存：")
    print("1. figures/training_loss_comparison.png")
    print("2. figures/test_accuracy_comparison.png")
    print("3. figures/training_time_comparison.png")
    print("4. figures/model_parameters_comparison.png")


# =========================
# 5. 保存实验结果到 CSV
# =========================
def save_results_to_csv(lstm_result, bilstm_result, epochs):
    """
    保存最终结果和每轮训练结果到 CSV 文件。
    """

    os.makedirs("results", exist_ok=True)

    # =========================
    # 1. 保存最终对比结果
    # =========================
    final_results = pd.DataFrame([
        {
            "Model": lstm_result["model_name"],
            "Parameters": lstm_result["params"],
            "Final_Test_Accuracy": lstm_result["final_test_acc"],
            "Final_Test_Accuracy_Percent": lstm_result["final_test_acc"] * 100,
            "Average_Epoch_Time": lstm_result["avg_epoch_time"],
            "Model_Save_Path": lstm_result["save_path"]
        },
        {
            "Model": bilstm_result["model_name"],
            "Parameters": bilstm_result["params"],
            "Final_Test_Accuracy": bilstm_result["final_test_acc"],
            "Final_Test_Accuracy_Percent": bilstm_result["final_test_acc"] * 100,
            "Average_Epoch_Time": bilstm_result["avg_epoch_time"],
            "Model_Save_Path": bilstm_result["save_path"]
        }
    ])

    final_results.to_csv(
        "results/final_results.csv",
        index=False,
        encoding="utf-8-sig"
    )

    # =========================
    # 2. 保存每个 epoch 的详细结果
    # =========================
    epoch_results = []

    for i in range(epochs):
        epoch_results.append({
            "Model": "LSTM",
            "Epoch": i + 1,
            "Train_Loss": lstm_result["train_losses"][i],
            "Train_Accuracy": lstm_result["train_accs"][i],
            "Train_Accuracy_Percent": lstm_result["train_accs"][i] * 100,
            "Test_Loss": lstm_result["test_losses"][i],
            "Test_Accuracy": lstm_result["test_accs"][i],
            "Test_Accuracy_Percent": lstm_result["test_accs"][i] * 100,
            "Epoch_Time": lstm_result["epoch_times"][i]
        })

        epoch_results.append({
            "Model": "Bi-LSTM",
            "Epoch": i + 1,
            "Train_Loss": bilstm_result["train_losses"][i],
            "Train_Accuracy": bilstm_result["train_accs"][i],
            "Train_Accuracy_Percent": bilstm_result["train_accs"][i] * 100,
            "Test_Loss": bilstm_result["test_losses"][i],
            "Test_Accuracy": bilstm_result["test_accs"][i],
            "Test_Accuracy_Percent": bilstm_result["test_accs"][i] * 100,
            "Epoch_Time": bilstm_result["epoch_times"][i]
        })

    epoch_results = pd.DataFrame(epoch_results)

    epoch_results.to_csv(
        "results/epoch_results.csv",
        index=False,
        encoding="utf-8-sig"
    )

    print("\nCSV 结果文件已保存：")
    print("1. results/final_results.csv")
    print("2. results/epoch_results.csv")


# =========================
# 6. 主函数
# =========================
def main():
    set_seed(42)

    # =========================
    # 实验超参数
    # =========================
    batch_size = 64

    # 如果电脑比较慢，可以先用 128；想完全按原实验可改回 256
    max_len = 256

    # 如果电脑比较慢，可以先用 10000；想完全按原实验可改回 20000
    max_vocab_size = 20000

    embed_dim = 128
    hidden_dim = 256
    num_layers = 1

    learning_rate = 1e-3

    # 电脑慢可以先跑 3 轮；想完全按课件要求可改回 5
    epochs = 5

    # =========================
    # 选择设备
    # =========================
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("当前使用设备:", device)

    # =========================
    # 加载数据
    # =========================
    train_loader, test_loader, vocab_size, word2idx = get_dataloaders(
        csv_path="data/IMDB Dataset.csv",
        batch_size=batch_size,
        max_len=max_len,
        max_vocab_size=max_vocab_size,
        min_freq=2,
        test_size=0.2,
        random_state=42
    )

    # =========================
    # 保存词表
    # =========================
    os.makedirs("checkpoints", exist_ok=True)

    with open("checkpoints/word2idx.json", "w", encoding="utf-8") as f:
        json.dump(word2idx, f, ensure_ascii=False, indent=2)

    print("词表已保存到: checkpoints/word2idx.json")

    print("词表大小:", vocab_size)
    print("训练 batch 数量:", len(train_loader))
    print("测试 batch 数量:", len(test_loader))

    # =========================
    # 训练单向 LSTM
    # =========================
    lstm_result = run_experiment(
        model_name="LSTM",
        bidirectional=False,
        train_loader=train_loader,
        test_loader=test_loader,
        vocab_size=vocab_size,
        device=device,
        epochs=epochs,
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        learning_rate=learning_rate
    )

    # =========================
    # 训练双向 Bi-LSTM
    # =========================
    bilstm_result = run_experiment(
        model_name="Bi-LSTM",
        bidirectional=True,
        train_loader=train_loader,
        test_loader=test_loader,
        vocab_size=vocab_size,
        device=device,
        epochs=epochs,
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        learning_rate=learning_rate
    )

    # =========================
    # 输出实验结果对比表
    # =========================
    print("\n" + "=" * 70)
    print("实验结果对比")
    print("=" * 70)

    print(
        f"{'模型':<12}"
        f"{'参数量':<15}"
        f"{'最终测试准确率(%)':<20}"
        f"{'平均每轮时间(s)':<18}"
    )

    print("-" * 70)

    print(
        f"{lstm_result['model_name']:<12}"
        f"{lstm_result['params']:<15}"
        f"{lstm_result['final_test_acc'] * 100:<20.2f}"
        f"{lstm_result['avg_epoch_time']:<18.2f}"
    )

    print(
        f"{bilstm_result['model_name']:<12}"
        f"{bilstm_result['params']:<15}"
        f"{bilstm_result['final_test_acc'] * 100:<20.2f}"
        f"{bilstm_result['avg_epoch_time']:<18.2f}"
    )

    print("=" * 70)

    # =========================
    # 保存 CSV 结果
    # =========================
    save_results_to_csv(
        lstm_result=lstm_result,
        bilstm_result=bilstm_result,
        epochs=epochs
    )

    # =========================
    # 自动生成文字结论
    # =========================
    if bilstm_result["final_test_acc"] > lstm_result["final_test_acc"]:
        print("结论1：Bi-LSTM 的最终测试准确率更高，说明双向结构能够利用更充分的上下文信息。")
    else:
        print("结论1：单向 LSTM 的最终测试准确率更高或与 Bi-LSTM 接近，说明当前实验设置下双向结构优势不明显。")

    if bilstm_result["avg_epoch_time"] > lstm_result["avg_epoch_time"]:
        print("结论2：Bi-LSTM 的平均每轮训练时间更长，说明双向结构带来了更高的计算开销。")
    else:
        print("结论2：Bi-LSTM 的平均训练时间没有明显增加。")

    if bilstm_result["params"] > lstm_result["params"]:
        print("结论3：Bi-LSTM 的参数量更多，因为它同时包含正向和反向两个 LSTM。")

    # =========================
    # 绘制可视化对比图
    # =========================
    plot_results(
        lstm_result=lstm_result,
        bilstm_result=bilstm_result,
        epochs=epochs
    )


if __name__ == "__main__":
    main()