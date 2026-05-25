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
output_dir = "results_cifar_unconditional"
checkpoint_dir = "checkpoints_cifar_unconditional"

batch_size = 64
num_epochs = 200
lr = 1e-4

noise_steps = 300
img_size = 32
img_channels = 3

sample_interval = 20
checkpoint_interval = 50

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =========================
# 2. 创建保存目录
# =========================
os.makedirs(output_dir, exist_ok=True)
os.makedirs(checkpoint_dir, exist_ok=True)


# =========================
# 3. 加载 CIFAR-10 数据
# =========================
train_loader = get_cifar10_loader(
    data_root=data_root,
    batch_size=batch_size,
    num_workers=2,
    train=True
)


# =========================
# 4. 创建扩散过程和无条件去噪模型
# =========================
diffusion = Diffusion(
    noise_steps=noise_steps,
    img_size=img_size,
    img_channels=img_channels,
    device=device
)

model = DenoiseUNet(
    image_channels=img_channels,
    conditional=False
).to(device)


# =========================
# 5. 损失函数和优化器
# =========================
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=lr)


loss_list = []
iteration_list = []
iteration = 0


# =========================
# 6. 开始训练
# =========================
print("Start training unconditional DDPM on CIFAR-10")
print("Using device:", device)
print("Noise steps:", noise_steps)
print("Epochs:", num_epochs)

for epoch in range(1, num_epochs + 1):

    model.train()

    for batch_idx, (images, _) in enumerate(train_loader):

        images = images.to(device)

        # 随机采样时间步
        t = diffusion.sample_timesteps(images.shape[0])

        # 前向扩散
        x_t, noise = diffusion.noise_images(images, t)

        # 预测噪声
        predicted_noise = model(x_t, t)

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
    # 7. 定期采样
    # =========================
    if epoch == 1 or epoch % sample_interval == 0:
        sampled_images = diffusion.sample(
            model=model,
            n=64,
            labels=None
        )

        save_image(
            sampled_images,
            f"{output_dir}/epoch_{epoch:03d}.png",
            normalize=True,
            nrow=8
        )

        print(f"Saved sampled images at epoch {epoch}")

    # =========================
    # 8. 定期保存模型
    # =========================
    if epoch % checkpoint_interval == 0:
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": loss.item(),
                "noise_steps": noise_steps
            },
            f"{checkpoint_dir}/ddpm_cifar_unconditional_epoch_{epoch}.pth"
        )

        print(f"Saved checkpoint at epoch {epoch}")


# =========================
# 9. 绘制 loss 曲线
# =========================
plt.figure(figsize=(10, 6))
plt.plot(iteration_list, loss_list, label="Noise Prediction Loss")
plt.xlabel("Iteration")
plt.ylabel("MSE Loss")
plt.title("Unconditional DDPM Training Loss on CIFAR-10")
plt.legend()
plt.grid(True)
plt.savefig(f"{output_dir}/loss_curve_unconditional.png", dpi=300)
plt.close()


# =========================
# 10. 保存最终模型
# =========================
torch.save(
    {
        "epoch": num_epochs,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "noise_steps": noise_steps
    },
    f"{checkpoint_dir}/ddpm_cifar_unconditional_final.pth"
)

print("Training finished.")