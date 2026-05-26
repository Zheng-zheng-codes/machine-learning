import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torchvision.utils import save_image

from dataset import get_mnist_loader
from model import DenoiseUNet
from diffusion import Diffusion


# =========================
# 1. 基本参数设置
# =========================
data_root = "../data"
output_dir = "results_unconditional"
checkpoint_dir = "checkpoints_unconditional"

batch_size = 128
num_epochs = 50
lr = 1e-4

noise_steps = 1000
img_size = 28
img_channels = 1

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =========================
# 2. 创建保存目录
# =========================
os.makedirs(output_dir, exist_ok=True)
os.makedirs(checkpoint_dir, exist_ok=True)


# =========================
# 3. 加载 MNIST 数据
# =========================
train_loader = get_mnist_loader(
    data_root=data_root,
    batch_size=batch_size,
    num_workers=2,
    train=True
)


# =========================
# 4. 创建扩散过程和去噪模型
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
print("Start training unconditional DDPM on MNIST")
print("Using device:", device)

for epoch in range(1, num_epochs + 1):

    for batch_idx, (images, _) in enumerate(train_loader):

        images = images.to(device)

        # 随机采样时间步
        t = diffusion.sample_timesteps(images.shape[0])

        # 给原图加噪声
        x_t, noise = diffusion.noise_images(images, t)

        # 模型预测噪声
        predicted_noise = model(x_t, t)

        # 预测噪声和真实噪声做 MSE
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
    # 7. 每轮采样生成图片
    # =========================
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

    # =========================
    # 8. 每 5 轮保存模型
    # =========================
    if epoch % 5 == 0:
        torch.save(
            model.state_dict(),
            f"{checkpoint_dir}/ddpm_unconditional_epoch_{epoch}.pth"
        )


# =========================
# 9. 绘制 loss 曲线
# =========================
plt.figure(figsize=(10, 6))
plt.plot(iteration_list, loss_list, label="Noise Prediction Loss")
plt.xlabel("Iteration")
plt.ylabel("MSE Loss")
plt.title("Unconditional DDPM Training Loss on MNIST")
plt.legend()
plt.grid(True)
plt.savefig(f"{output_dir}/loss_curve_unconditional.png", dpi=300)
plt.close()

print("Training finished.")