import torch
import torch.nn as nn


# CGAN 生成器
# 输入:
#   noise:  [batch_size, 100, 1, 1]
#   labels: [batch_size]
# 输出:
#   fake_images: [batch_size, 1, 28, 28]
class ConditionalGenerator(nn.Module):

    def __init__(
        self,
        noise_dim=100,
        num_classes=10,
        image_channels=1,
        feature_maps=64,
        embed_dim=100
    ):
        super(ConditionalGenerator, self).__init__()

        self.noise_dim = noise_dim
        self.num_classes = num_classes
        self.embed_dim = embed_dim

        # 把类别标签变成向量
        self.label_embedding = nn.Embedding(num_classes, embed_dim)

        # noise + label_embedding 合并后作为生成器输入
        input_dim = noise_dim + embed_dim

        self.main = nn.Sequential(
            # 输入: [batch_size, input_dim, 1, 1]
            # 输出: [batch_size, 256, 7, 7]
            nn.ConvTranspose2d(
                in_channels=input_dim,
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

            # 输出范围为 [-1, 1]
            nn.Tanh()
        )

    def forward(self, noise, labels):
        label_embed = self.label_embedding(labels)
        # [batch_size, embed_dim]

        label_embed = label_embed.view(label_embed.size(0), self.embed_dim, 1, 1)
        # [batch_size, embed_dim, 1, 1]

        x = torch.cat([noise, label_embed], dim=1)
        # [batch_size, noise_dim + embed_dim, 1, 1]

        fake_images = self.main(x)

        return fake_images


# CGAN 判别器
# 输入:
#   images: [batch_size, 1, 28, 28]
#   labels: [batch_size]
# 输出:
#   prob:   [batch_size]
class ConditionalDiscriminator(nn.Module):

    def __init__(
        self,
        num_classes=10,
        image_channels=1,
        feature_maps=64,
        image_size=28
    ):
        super(ConditionalDiscriminator, self).__init__()

        self.num_classes = num_classes
        self.image_size = image_size

        # 把类别标签变成一张 28×28 的条件图
        self.label_embedding = nn.Embedding(
            num_classes,
            image_size * image_size
        )

        # 原图是 1 通道，标签条件图是 1 通道
        # 拼接后输入通道数变成 2
        input_channels = image_channels + 1

        self.main = nn.Sequential(
            # 输入: [batch_size, 2, 28, 28]
            # 输出: [batch_size, 64, 14, 14]
            nn.Conv2d(
                in_channels=input_channels,
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
            nn.BatchNorm2d(feature_maps * 2),
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
            ),

            nn.Sigmoid()
        )

    def forward(self, images, labels):
        label_embed = self.label_embedding(labels)
        # [batch_size, 28*28]

        label_map = label_embed.view(
            label_embed.size(0),
            1,
            self.image_size,
            self.image_size
        )
        # [batch_size, 1, 28, 28]

        x = torch.cat([images, label_map], dim=1)
        # [batch_size, 2, 28, 28]

        prob = self.main(x)

        return prob.view(-1)


# 权重初始化函数
def weights_init(model):

    classname = model.__class__.__name__

    if classname.find("Conv") != -1:
        nn.init.normal_(model.weight.data, 0.0, 0.02)

    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(model.weight.data, 1.0, 0.02)
        nn.init.constant_(model.bias.data, 0)


# 测试模型输入输出形状
if __name__ == "__main__":
    batch_size = 8
    noise_dim = 100
    num_classes = 10

    netG = ConditionalGenerator(
        noise_dim=noise_dim,
        num_classes=num_classes
    )

    netD = ConditionalDiscriminator(
        num_classes=num_classes
    )

    netG.apply(weights_init)
    netD.apply(weights_init)

    noise = torch.randn(batch_size, noise_dim, 1, 1)
    labels = torch.randint(0, num_classes, (batch_size,))

    fake_images = netG(noise, labels)
    output = netD(fake_images, labels)

    print("noise 形状:", noise.shape)
    print("labels 形状:", labels.shape)
    print("fake_images 形状:", fake_images.shape)
    print("Discriminator 输出形状:", output.shape)
    print("labels:", labels)