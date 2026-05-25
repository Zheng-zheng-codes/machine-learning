import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torchvision.utils import save_image

from dataset import get_mnist_loader
from CGAN import (
    ConditionalGenerator,
    ConditionalDiscriminator,
    weights_init
)


# =========================
# 1. 基本参数设置
# =========================
data_root = "../data"
output_dir = "results_cgan"
checkpoint_dir = "checkpoints_cgan"

batch_size = 128
noise_dim = 100
num_classes = 10
num_epochs = 100

lr = 0.0002
beta1 = 0.5

# 每个类别每轮保存多少张图片
samples_per_class = 8

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =========================
# 2. MNIST 类别名
# =========================
class_names = [
    "digit_0", "digit_1", "digit_2", "digit_3", "digit_4",
    "digit_5", "digit_6", "digit_7", "digit_8", "digit_9"
]


# =========================
# 3. 创建保存目录
# =========================
os.makedirs(output_dir, exist_ok=True)
os.makedirs(checkpoint_dir, exist_ok=True)


# =========================
# 4. 加载 MNIST 数据
# =========================
train_loader = get_mnist_loader(
    data_root=data_root,
    batch_size=batch_size,
    num_workers=2,
    train=True
)


# =========================
# 5. 创建生成器和判别器
# =========================
netG = ConditionalGenerator(
    noise_dim=noise_dim,
    num_classes=num_classes
).to(device)

netD = ConditionalDiscriminator(
    num_classes=num_classes
).to(device)

netG.apply(weights_init)
netD.apply(weights_init)


# =========================
# 6. 定义损失函数和优化器
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


# =========================
# 7. 固定噪声和固定标签
# 用来每轮生成同一组条件样本，方便对比变化
# =========================
fixed_noise = torch.randn(
    num_classes * samples_per_class,
    noise_dim,
    1,
    1,
    device=device
)

fixed_labels = torch.arange(
    num_classes,
    device=device
).repeat_interleave(samples_per_class)


# =========================
# 8. 按类别保存生成图片
# =========================
def save_images_by_class(generator, fixed_noise, fixed_labels, epoch, output_dir):
    """
    每个 epoch 按类别分别保存生成图片。

    保存结构：
    results_cgan/
    ├── epoch_001/
    │   ├── class_0_digit_0/
    │   │   ├── sample_01.png
    │   │   └── ...
    │   ├── class_1_digit_1/
    │   └── ...
    """

    generator.eval()

    epoch_dir = os.path.join(output_dir, f"epoch_{epoch:03d}")
    os.makedirs(epoch_dir, exist_ok=True)

    with torch.no_grad():
        fake_images = generator(fixed_noise, fixed_labels).detach().cpu()

    for class_idx in range(num_classes):
        class_dir = os.path.join(
            epoch_dir,
            f"class_{class_idx}_{class_names[class_idx]}"
        )
        os.makedirs(class_dir, exist_ok=True)

        start = class_idx * samples_per_class
        end = start + samples_per_class

        class_images = fake_images[start:end]

        for i, img in enumerate(class_images):
            save_image(
                img,
                os.path.join(class_dir, f"sample_{i + 1:02d}.png"),
                normalize=True
            )

    generator.train()


# =========================
# 9. 记录 loss
# =========================
lossD_list = []
lossG_list = []
iteration_list = []
iteration = 0


# =========================
# 10. 开始训练
# =========================
print("Start training CGAN on MNIST")
print("Using device:", device)

for epoch in range(1, num_epochs + 1):

    for batch_idx, (real_images, labels) in enumerate(train_loader):

        real_images = real_images.to(device)
        labels = labels.to(device)

        current_batch_size = real_images.size(0)

        real_targets = torch.ones(current_batch_size, device=device)
        fake_targets = torch.zeros(current_batch_size, device=device)

        # =========================
        # 10.1 训练判别器 Discriminator
        # =========================
        netD.zero_grad()

        # 真实图片 + 正确标签，希望判别器输出 1
        output_real = netD(real_images, labels)
        lossD_real = criterion(output_real, real_targets)

        # 随机噪声 + 当前 batch 的标签，生成对应类别假图
        noise = torch.randn(
            current_batch_size,
            noise_dim,
            1,
            1,
            device=device
        )

        fake_images = netG(noise, labels)

        # 假图片 + 对应标签，希望判别器输出 0
        output_fake = netD(fake_images.detach(), labels)
        lossD_fake = criterion(output_fake, fake_targets)

        # 判别器总损失
        lossD = lossD_real + lossD_fake
        lossD.backward()
        optimizerD.step()

        # =========================
        # 10.2 训练生成器 Generator
        # =========================
        netG.zero_grad()

        # 生成器希望 fake_images + labels 被判别器认为是真的
        output_for_G = netD(fake_images, labels)
        lossG = criterion(output_for_G, real_targets)

        lossG.backward()
        optimizerG.step()

        # =========================
        # 10.3 记录 loss
        # =========================
        iteration += 1
        lossD_list.append(lossD.item())
        lossG_list.append(lossG.item())
        iteration_list.append(iteration)

        # =========================
        # 10.4 打印训练信息
        # =========================
        if batch_idx % 100 == 0:
            print(
                f"Epoch [{epoch}/{num_epochs}] "
                f"Batch [{batch_idx}/{len(train_loader)}] "
                f"Loss_D: {lossD.item():.4f} "
                f"Loss_G: {lossG.item():.4f}"
            )

    # =========================
    # 11. 每轮保存总览图
    # =========================
    with torch.no_grad():
        fake_samples = netG(fixed_noise, fixed_labels).detach().cpu()

    save_image(
        fake_samples,
        f"{output_dir}/epoch_{epoch:03d}.png",
        normalize=True,
        nrow=samples_per_class
    )

    # =========================
    # 12. 每轮按类别保存
    # =========================
    save_images_by_class(
        generator=netG,
        fixed_noise=fixed_noise,
        fixed_labels=fixed_labels,
        epoch=epoch,
        output_dir=output_dir
    )

    # =========================
    # 13. 每 5 轮保存一次模型
    # =========================
    if epoch % 5 == 0:
        torch.save(netG.state_dict(), f"{checkpoint_dir}/netG_epoch_{epoch}.pth")
        torch.save(netD.state_dict(), f"{checkpoint_dir}/netD_epoch_{epoch}.pth")


# =========================
# 14. 绘制 loss 曲线
# =========================
plt.figure(figsize=(10, 6))

plt.plot(iteration_list, lossD_list, label="Discriminator Loss")
plt.plot(iteration_list, lossG_list, label="Generator Loss")

plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.title("CGAN Training Loss on MNIST")
plt.legend()
plt.grid(True)

plt.savefig(f"{output_dir}/loss_curve_cgan.png", dpi=300)
plt.close()

print("Training finished.")