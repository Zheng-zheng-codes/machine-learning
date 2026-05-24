import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torchvision.utils import save_image

from dataset import get_cifar10_loader
from GAN import Generator, Discriminator, weights_init


# =========================
# 1. 基本参数设置
# =========================
data_root = "../data"
output_dir = "results_gan"
checkpoint_dir = "checkpoints_gan"

batch_size = 128
noise_dim = 100
num_epochs = 100
lr = 0.0002
beta1 = 0.5

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
# 4. 创建生成器和判别器
# =========================
netG = Generator(noise_dim=noise_dim).to(device)
netD = Discriminator().to(device)

netG.apply(weights_init)
netD.apply(weights_init)


# =========================
# 5. 定义损失函数和优化器
# =========================
criterion = nn.BCELoss()

optimizerD = optim.Adam(
    netD.parameters(),
    lr=lr,
    betas=(beta1, 0.999)
)

optimizerG = optim.Adam(
    netG.parameters(),
    lr=lr,
    betas=(beta1, 0.999)
)


# 固定一组噪声，用来观察每一轮生成结果的变化
fixed_noise = torch.randn(64, noise_dim, 1, 1, device=device)
lossD_list = []
lossG_list = []
iteration_list = []
iteration = 0

# =========================
# 6. 开始训练
# =========================
print("Start training GAN on CIFAR-10")
print("Using device:", device)

for epoch in range(1, num_epochs + 1):

    for batch_idx, (real_images, _) in enumerate(train_loader):

        real_images = real_images.to(device)
        current_batch_size = real_images.size(0)

        real_labels = torch.ones(current_batch_size, device=device)
        fake_labels = torch.zeros(current_batch_size, device=device)

        # =========================
        # 6.1 训练判别器 Discriminator
        # =========================
        netD.zero_grad()

        # 真实图片，希望判别器输出 1
        output_real = netD(real_images)
        lossD_real = criterion(output_real, real_labels)

        # 生成假图片
        noise = torch.randn(current_batch_size, noise_dim, 1, 1, device=device)
        fake_images = netG(noise)

        # 假图片，希望判别器输出 0
        output_fake = netD(fake_images.detach())
        lossD_fake = criterion(output_fake, fake_labels)

        # 判别器总损失
        lossD = lossD_real + lossD_fake
        lossD.backward()
        optimizerD.step()

        # =========================
        # 6.2 训练生成器 Generator
        # =========================
        netG.zero_grad()

        # 生成器希望判别器把假图判断为真，所以标签用 real_labels
        output_for_G = netD(fake_images)
        lossG = criterion(output_for_G, real_labels)

        lossG.backward()
        optimizerG.step()

        iteration += 1
        lossD_list.append(lossD.item())
        lossG_list.append(lossG.item())
        iteration_list.append(iteration)

        # =========================
        # 6.3 打印训练信息
        # =========================
        if batch_idx % 100 == 0:
            print(
                f"Epoch [{epoch}/{num_epochs}] "
                f"Batch [{batch_idx}/{len(train_loader)}] "
                f"Loss_D: {lossD.item():.4f} "
                f"Loss_G: {lossG.item():.4f}"
            )

    # =========================
    # 7. 每轮保存生成图片
    # =========================
    with torch.no_grad():
        fake_samples = netG(fixed_noise).detach().cpu()

    save_image(
        fake_samples,
        f"{output_dir}/epoch_{epoch:03d}.png",
        normalize=True,
        nrow=8
    )

    # =========================
    # 8. 每 5 轮保存一次模型
    # =========================
    if epoch % 5 == 0:
        torch.save(netG.state_dict(), f"{checkpoint_dir}/netG_epoch_{epoch}.pth")
        torch.save(netD.state_dict(), f"{checkpoint_dir}/netD_epoch_{epoch}.pth")

# =========================
# 9. 绘制 loss 曲线
# =========================
plt.figure(figsize=(10, 6))

plt.plot(iteration_list, lossD_list, label="Discriminator Loss")
plt.plot(iteration_list, lossG_list, label="Generator Loss")

plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.title("GAN Training Loss on CIFAR-10")
plt.legend()
plt.grid(True)

plt.savefig(f"{output_dir}/loss_curve_gan.png", dpi=300)
plt.close()

print("Training finished.")
