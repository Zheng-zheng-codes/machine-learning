import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torchvision.utils import save_image

from dataset import get_cifar10_loader
from model import DenoiseUNet
from diffusion import Diffusion


# =========================
# 1. 基本参数设置
# =========================
data_root = "../data"
output_dir = "results_cifar_conditional"
checkpoint_dir = "checkpoints_cifar_conditional"

batch_size = 64
num_epochs = 100
lr = 1e-4

noise_steps = 1000
img_size = 32
img_channels = 3

num_classes = 10
samples_per_class = 4

sample_interval = 20
checkpoint_interval = 20

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =========================
# 2. CIFAR-10 类别名
# =========================
class_names = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]


# =========================
# 3. 创建保存目录
# =========================
os.makedirs(output_dir, exist_ok=True)
os.makedirs(checkpoint_dir, exist_ok=True)


# =========================
# 4. 加载 CIFAR-10 数据
# =========================
train_loader = get_cifar10_loader(
    data_root=data_root,
    batch_size=batch_size,
    num_workers=2,
    train=True
)


# =========================
# 5. 创建扩散过程和条件去噪模型
# =========================
diffusion = Diffusion(
    noise_steps=noise_steps,
    img_size=img_size,
    img_channels=img_channels,
    device=device
)

model = DenoiseUNet(
    image_channels=img_channels,
    num_classes=num_classes,
    conditional=True
).to(device)


# =========================
# 6. 损失函数和优化器
# =========================
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=lr)


# =========================
# 7. 固定标签
# 生成顺序：
# 0 0 0 0, 1 1 1 1, ..., 9 9 9 9
# =========================
fixed_labels = torch.arange(
    num_classes,
    device=device
).repeat_interleave(samples_per_class)


# =========================
# 8. 按类别保存生成图片
# 目录结构：
# results_cifar_conditional/
# ├── epoch_020.png
# ├── epoch_020/
# │   ├── class_0_airplane/
# │   │   ├── sample_01.png
# │   │   └── ...
# │   ├── class_1_automobile/
# │   └── ...
# =========================
def save_images_by_class(model, diffusion, epoch, output_dir):
    model.eval()

    epoch_dir = os.path.join(output_dir, f"epoch_{epoch:03d}")
    os.makedirs(epoch_dir, exist_ok=True)

    with torch.no_grad():
        sampled_images = diffusion.sample(
            model=model,
            n=num_classes * samples_per_class,
            labels=fixed_labels
        ).detach().cpu()

    # =========================
    # 8.1 保存总览图
    # =========================
    save_image(
        sampled_images,
        os.path.join(output_dir, f"epoch_{epoch:03d}.png"),
        normalize=True,
        nrow=samples_per_class
    )

    # =========================
    # 8.2 按类别分别保存
    # =========================
    for class_idx in range(num_classes):
        class_dir = os.path.join(
            epoch_dir,
            f"class_{class_idx}_{class_names[class_idx]}"
        )
        os.makedirs(class_dir, exist_ok=True)

        start = class_idx * samples_per_class
        end = start + samples_per_class

        class_images = sampled_images[start:end]

        for i, img in enumerate(class_images):
            save_image(
                img,
                os.path.join(class_dir, f"sample_{i + 1:02d}.png"),
                normalize=True
            )

    model.train()


# =========================
# 9. 记录 loss
# =========================
loss_list = []
iteration_list = []
iteration = 0


# =========================
# 10. 开始训练
# =========================
print("Start training conditional DDPM on CIFAR-10")
print("Using device:", device)
print("Noise steps:", noise_steps)
print("Epochs:", num_epochs)
print("Samples per class:", samples_per_class)

for epoch in range(1, num_epochs + 1):

    model.train()

    for batch_idx, (images, labels) in enumerate(train_loader):

        images = images.to(device)
        labels = labels.to(device)

        # 随机采样时间步
        t = diffusion.sample_timesteps(images.shape[0])

        # 前向扩散：x0 -> xt
        x_t, noise = diffusion.noise_images(images, t)

        # 条件扩散模型：输入 xt、t、label，预测噪声
        predicted_noise = model(x_t, t, labels)

        # MSE loss
        loss = criterion(predicted_noise, noise)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        iteration += 1
        loss_list.append(loss.item())
        iteration_list.append(iteration)

        if batch_idx % 100 == 0:
            print(
                f"Epoch [{epoch}/{num_epochs}] "
                f"Batch [{batch_idx}/{len(train_loader)}] "
                f"Loss: {loss.item():.6f}"
            )

    # =========================
    # 11. 定期按类别采样
    # =========================
    if epoch == 1 or epoch % sample_interval == 0:
        save_images_by_class(
            model=model,
            diffusion=diffusion,
            epoch=epoch,
            output_dir=output_dir
        )

        print(f"Saved conditional sampled images at epoch {epoch}")

    # =========================
    # 12. 定期保存模型
    # =========================
    if epoch % checkpoint_interval == 0:
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": loss.item(),
                "noise_steps": noise_steps,
                "num_classes": num_classes,
                "samples_per_class": samples_per_class,
                "class_names": class_names
            },
            f"{checkpoint_dir}/ddpm_cifar_conditional_epoch_{epoch}.pth"
        )

        print(f"Saved checkpoint at epoch {epoch}")


# =========================
# 13. 绘制 loss 曲线
# =========================
plt.figure(figsize=(10, 6))
plt.plot(iteration_list, loss_list, label="Noise Prediction Loss")
plt.xlabel("Iteration")
plt.ylabel("MSE Loss")
plt.title("Conditional DDPM Training Loss on CIFAR-10")
plt.legend()
plt.grid(True)
plt.savefig(f"{output_dir}/loss_curve_conditional.png", dpi=300)
plt.close()


# =========================
# 14. 保存最终模型
# =========================
torch.save(
    {
        "epoch": num_epochs,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "noise_steps": noise_steps,
        "num_classes": num_classes,
        "samples_per_class": samples_per_class,
        "class_names": class_names
    },
    f"{checkpoint_dir}/ddpm_cifar_conditional_final.pth"
)

print("Training finished.")