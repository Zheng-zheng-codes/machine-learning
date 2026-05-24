import torch
import torch.nn as nn

# 生成器
# 输入: [batch_size, 100, 1, 1]
# 输出: [batch_size, 3, 32, 32]
class Generator(nn.Module):

    def __init__(self, noise_dim=100, image_channels=3, feature_maps=64):
        super(Generator, self).__init__()

        self.main = nn.Sequential(
            # 输入: [batch_size, 100, 1, 1]
            # 输出: [batch_size, 256, 4, 4]
            nn.ConvTranspose2d(
                in_channels=noise_dim,
                out_channels=feature_maps * 4,
                kernel_size=4,
                stride=1,
                padding=0,
                bias=False
            ),
            nn.BatchNorm2d(feature_maps * 4),
            nn.ReLU(True),

            # 输入: [batch_size, 256, 4, 4]
            # 输出: [batch_size, 128, 8, 8]
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

            # 输入: [batch_size, 128, 8, 8]
            # 输出: [batch_size, 64, 16, 16]
            nn.ConvTranspose2d(
                in_channels=feature_maps * 2,
                out_channels=feature_maps,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(feature_maps),
            nn.ReLU(True),

            # 输入: [batch_size, 64, 16, 16]
            # 输出: [batch_size, 3, 32, 32]
            nn.ConvTranspose2d(
                in_channels=feature_maps,
                out_channels=image_channels,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False
            ),

            # 输出范围变成 [-1, 1]
            nn.Tanh()
        )

    def forward(self, noise):
        fake_images = self.main(noise)
        return fake_images

# 判别器
# 输入: [batch_size, 3, 32, 32]
# 输出: [batch_size]
class Discriminator(nn.Module):

    def __init__(self, image_channels=3, feature_maps=64):
        super(Discriminator, self).__init__()

        self.main = nn.Sequential(
            # 输入: [batch_size, 3, 32, 32]
            # 输出: [batch_size, 64, 16, 16]
            nn.Conv2d(
                in_channels=image_channels,
                out_channels=feature_maps,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False
            ),
            nn.LeakyReLU(0.2, inplace=True),

            # 输入: [batch_size, 64, 16, 16]
            # 输出: [batch_size, 128, 8, 8]
            nn.Conv2d(
                in_channels=feature_maps,
                out_channels=feature_maps * 2,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(feature_maps * 2),
            nn.LeakyReLU(0.2, inplace=True),

            # 输入: [batch_size, 128, 8, 8]
            # 输出: [batch_size, 256, 4, 4]
            nn.Conv2d(
                in_channels=feature_maps * 2,
                out_channels=feature_maps * 4,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(feature_maps * 4),
            nn.LeakyReLU(0.2, inplace=True),

            # 输入: [batch_size, 256, 4, 4]
            # 输出: [batch_size, 1, 1, 1]
            nn.Conv2d(
                in_channels=feature_maps * 4,
                out_channels=1,
                kernel_size=4,
                stride=1,
                padding=0,
                bias=False
            ),

            # 输出真假概率
            nn.Sigmoid()
        )

    def forward(self, images):
        prob = self.main(images)
        return prob.view(-1)

# 权重初始化函数
def weights_init(model):

    classname = model.__class__.__name__

    if classname.find("Conv") != -1:
        nn.init.normal_(model.weight.data, 0.0, 0.02)

    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(model.weight.data, 1.0, 0.02)
        nn.init.constant_(model.bias.data, 0)


if __name__ == "__main__":
    batch_size = 8
    noise_dim = 100

    netG = Generator(noise_dim=noise_dim)
    netD = Discriminator()

    netG.apply(weights_init)
    netD.apply(weights_init)

    noise = torch.randn(batch_size, noise_dim, 1, 1)

    fake_images = netG(noise)
    output = netD(fake_images)

    print("Generator 输入形状:", noise.shape)
    print("Generator 输出形状:", fake_images.shape)
    print("Discriminator 输出形状:", output.shape)