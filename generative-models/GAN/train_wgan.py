import os
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from torchvision.utils import save_image

from dataset import get_cifar10_loader
from WGAN import Generator, Critic, weights_init


# =========================
# 1. 基本参数设置
# =========================
data_root = "../data"
output_dir = "results_wgan"
checkpoint_dir = "checkpoints_wgan"

batch_size = 128
noise_dim = 100
num_epochs = 100

lr = 0.0002
beta1 = 0.5
beta2 = 0.999

critic_iters = 5
lambda_gp = 10

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
# 4. 创建生成器和评价器
# =========================
netG = Generator(noise_dim=noise_dim).to(device)
critic = Critic().to(device)

netG.apply(weights_init)
critic.apply(weights_init)


# =========================
# 5. 定义优化器
# =========================
optimizerG = optim.Adam(
    netG.parameters(),
    lr=lr,
    betas=(beta1, beta2)
)

optimizerC = optim.Adam(
    critic.parameters(),
    lr=lr,
    betas=(beta1, beta2)
)


# 固定噪声，用于观察每一轮生成结果变化
fixed_noise = torch.randn(64, noise_dim, 1, 1, device=device)


# =========================
# 6. 梯度惩罚函数
# =========================
def gradient_penalty(critic, real_images, fake_images, device):
    batch_size = real_images.size(0)

    # epsilon 用来在真实图和假图之间随机插值
    epsilon = torch.rand(batch_size, 1, 1, 1, device=device)

    interpolated_images = epsilon * real_images + (1 - epsilon) * fake_images
    interpolated_images.requires_grad_(True)

    mixed_scores = critic(interpolated_images)

    gradients = torch.autograd.grad(
        outputs=mixed_scores,
        inputs=interpolated_images,
        grad_outputs=torch.ones_like(mixed_scores),
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]

    gradients = gradients.view(batch_size, -1)

    gradient_norm = gradients.norm(2, dim=1)

    gp = ((gradient_norm - 1) ** 2).mean()

    return gp


# =========================
# 7. 记录 loss
# =========================
lossC_list = []
lossG_list = []
iteration_list = []
iteration = 0


# =========================
# 8. 开始训练
# =========================
print("Start training WGAN-GP on CIFAR-10")
print("Using device:", device)

for epoch in range(1, num_epochs + 1):

    for batch_idx, (real_images, _) in enumerate(train_loader):

        real_images = real_images.to(device)
        current_batch_size = real_images.size(0)

        # =========================
        # 8.1 训练 Critic
        # =========================
        critic.zero_grad()

        noise = torch.randn(current_batch_size, noise_dim, 1, 1, device=device)
        fake_images = netG(noise)

        real_scores = critic(real_images)
        fake_scores = critic(fake_images.detach())

        gp = gradient_penalty(
            critic=critic,
            real_images=real_images,
            fake_images=fake_images.detach(),
            device=device
        )

        lossC = fake_scores.mean() - real_scores.mean() + lambda_gp * gp

        lossC.backward()
        optimizerC.step()

        # =========================
        # 8.2 每训练多次 Critic，再训练一次 Generator
        # =========================
        if batch_idx % critic_iters == 0:

            netG.zero_grad()

            noise = torch.randn(current_batch_size, noise_dim, 1, 1, device=device)
            fake_images = netG(noise)

            fake_scores_for_G = critic(fake_images)

            lossG = -fake_scores_for_G.mean()

            lossG.backward()
            optimizerG.step()

            iteration += 1
            lossC_list.append(lossC.item())
            lossG_list.append(lossG.item())
            iteration_list.append(iteration)

        # =========================
        # 8.3 打印训练信息
        # =========================
        if batch_idx % 100 == 0:
            print(
                f"Epoch [{epoch}/{num_epochs}] "
                f"Batch [{batch_idx}/{len(train_loader)}] "
                f"Loss_C: {lossC.item():.4f} "
                f"Loss_G: {lossG.item():.4f} "
                f"GP: {gp.item():.4f}"
            )

    # =========================
    # 9. 每轮保存生成图片
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
    # 10. 每 5 轮保存一次模型
    # =========================
    if epoch % 5 == 0:
        torch.save(netG.state_dict(), f"{checkpoint_dir}/netG_epoch_{epoch}.pth")
        torch.save(critic.state_dict(), f"{checkpoint_dir}/critic_epoch_{epoch}.pth")


# =========================
# 11. 绘制 loss 曲线
# =========================
plt.figure(figsize=(10, 6))

plt.plot(iteration_list, lossC_list, label="Critic Loss")
plt.plot(iteration_list, lossG_list, label="Generator Loss")

plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.title("WGAN-GP Training Loss on CIFAR-10")
plt.legend()
plt.grid(True)

plt.savefig(f"{output_dir}/loss_curve_wgan_gp.png", dpi=300)
plt.close()

print("Training finished.")