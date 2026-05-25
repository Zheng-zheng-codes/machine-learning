import torch
import torch.nn as nn


# WGAN 生成器
# 输入: [batch_size, 100, 1, 1]
# 输出: [batch_size, 1, 28, 28]
class Generator(nn.Module):

    def __init__(self, noise_dim=100, image_channels=1, feature_maps=64):
        super(Generator, self).__init__()

        self.main = nn.Sequential(
            # 输入: [batch_size, 100, 1, 1]
            # 输出: [batch_size, 256, 7, 7]
            nn.ConvTranspose2d(
                in_channels=noise_dim,
                out_channels=feature_maps * 4,
                kernel_size=7,
                stride=1,
                padding=0,
                bias=False
            ),
            nn.BatchNorm2d(feature_maps * 4),
            nn.ReLU(True),

            # 输入: [batch_size, 256, 7, 7]
            # 输出: [batch_size, 128, 14, 14]
            nn.ConvTranspose2d(
                in_channels=feature_maps * 4,
                out_channels=feature_maps * 2,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(feature_maps * 2),
            nn.ReLU(True),

            # 输入: [batch_size, 128, 14, 14]
            # 输出: [batch_size, 1, 28, 28]
            nn.ConvTranspose2d(
                in_channels=feature_maps * 2,
                out_channels=image_channels,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False
            ),

            # 输出范围仍然是 [-1, 1]
            nn.Tanh()
        )

    def forward(self, noise):
        fake_images = self.main(noise)
        return fake_images


# WGAN 评价器 Critic
# 输入: [batch_size, 1, 28, 28]
# 输出: [batch_size]
class Critic(nn.Module):

    def __init__(self, image_channels=1, feature_maps=64):
        super(Critic, self).__init__()

        self.main = nn.Sequential(
            # 输入: [batch_size, 1, 28, 28]
            # 输出: [batch_size, 64, 14, 14]
            nn.Conv2d(
                in_channels=image_channels,
                out_channels=feature_maps,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False
            ),
            nn.LeakyReLU(0.2, inplace=True),

            # 输入: [batch_size, 64, 14, 14]
            # 输出: [batch_size, 128, 7, 7]
            nn.Conv2d(
                in_channels=feature_maps,
                out_channels=feature_maps * 2,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False
            ),
            nn.InstanceNorm2d(feature_maps * 2, affine=True),
            nn.LeakyReLU(0.2, inplace=True),

            # 输入: [batch_size, 128, 7, 7]
            # 输出: [batch_size, 1, 1, 1]
            nn.Conv2d(
                in_channels=feature_maps * 2,
                out_channels=1,
                kernel_size=7,
                stride=1,
                padding=0,
                bias=False
            )

            # 注意：WGAN / WGAN-GP 这里没有 Sigmoid
        )

    def forward(self, images):
        scores = self.main(images)
        return scores.view(-1)


# 权重初始化函数
def weights_init(model):

    classname = model.__class__.__name__

    if classname.find("Conv") != -1:
        nn.init.normal_(model.weight.data, 0.0, 0.02)

    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(model.weight.data, 1.0, 0.02)
        nn.init.constant_(model.bias.data, 0)

    elif classname.find("InstanceNorm") != -1:
        if model.weight is not None:
            nn.init.normal_(model.weight.data, 1.0, 0.02)
        if model.bias is not None:
            nn.init.constant_(model.bias.data, 0)


if __name__ == "__main__":
    batch_size = 8
    noise_dim = 100

    netG = Generator(noise_dim=noise_dim)
    critic = Critic()

    netG.apply(weights_init)
    critic.apply(weights_init)

    noise = torch.randn(batch_size, noise_dim, 1, 1)

    fake_images = netG(noise)
    scores = critic(fake_images)

    print("Generator 输入形状:", noise.shape)
    print("Generator 输出形状:", fake_images.shape)
    print("Critic 输出形状:", scores.shape)
    print("Critic 输出示例:", scores)