import torch
import torch.nn as nn
import math


# =========================
# 1. 时间步编码
# =========================
class SinusoidalPositionEmbeddings(nn.Module):
    """
    把时间步 t 编码成一个向量。
    扩散模型需要知道当前处于第几步噪声。
    """

    def __init__(self, dim):
        super(SinusoidalPositionEmbeddings, self).__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2

        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(
            torch.arange(half_dim, device=device) * -embeddings
        )

        embeddings = time[:, None] * embeddings[None, :]

        embeddings = torch.cat(
            (embeddings.sin(), embeddings.cos()),
            dim=-1
        )

        return embeddings


# =========================
# 2. 基础卷积模块
# =========================
class ConvBlock(nn.Module):
    """
    一个带时间/标签条件的卷积块。
    """

    def __init__(self, in_channels, out_channels, emb_dim):
        super(ConvBlock, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1
        )

        self.bn1 = nn.BatchNorm2d(out_channels)
        self.act1 = nn.ReLU(inplace=True)

        self.emb_layer = nn.Linear(emb_dim, out_channels)

        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1
        )

        self.bn2 = nn.BatchNorm2d(out_channels)
        self.act2 = nn.ReLU(inplace=True)

    def forward(self, x, emb):
        h = self.conv1(x)
        h = self.bn1(h)
        h = self.act1(h)

        emb_out = self.emb_layer(emb)
        emb_out = emb_out[:, :, None, None]

        h = h + emb_out

        h = self.conv2(h)
        h = self.bn2(h)
        h = self.act2(h)

        return h


# =========================
# 3. 简化版 U-Net 去噪网络
# =========================
class DenoiseUNet(nn.Module):
    """
    CIFAR-10 扩散模型用的简化 U-Net。

    输入:
        x:      加噪图片 [batch_size, 3, 32, 32]
        t:      时间步 [batch_size]
        labels: 类别标签 [batch_size]，条件模型使用

    输出:
        predicted_noise: 预测噪声 [batch_size, 3, 32, 32]
    """

    def __init__(
        self,
        image_channels=3,
        base_channels=64,
        time_dim=128,
        num_classes=10,
        conditional=True
    ):
        super(DenoiseUNet, self).__init__()

        self.conditional = conditional
        self.time_dim = time_dim

        # 时间步编码
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.ReLU(inplace=True),
            nn.Linear(time_dim, time_dim)
        )

        # 类别标签编码
        if self.conditional:
            self.label_embedding = nn.Embedding(num_classes, time_dim)

        # Encoder: 32 -> 16 -> 8
        self.inc = ConvBlock(
            image_channels,
            base_channels,
            time_dim
        )

        self.down1 = nn.Conv2d(
            base_channels,
            base_channels * 2,
            kernel_size=4,
            stride=2,
            padding=1
        )

        self.block1 = ConvBlock(
            base_channels * 2,
            base_channels * 2,
            time_dim
        )

        self.down2 = nn.Conv2d(
            base_channels * 2,
            base_channels * 4,
            kernel_size=4,
            stride=2,
            padding=1
        )

        self.block2 = ConvBlock(
            base_channels * 4,
            base_channels * 4,
            time_dim
        )

        # Bottleneck
        self.bot = ConvBlock(
            base_channels * 4,
            base_channels * 4,
            time_dim
        )

        # Decoder: 8 -> 16 -> 32
        self.up1 = nn.ConvTranspose2d(
            base_channels * 4,
            base_channels * 2,
            kernel_size=4,
            stride=2,
            padding=1
        )

        self.block_up1 = ConvBlock(
            base_channels * 4,
            base_channels * 2,
            time_dim
        )

        self.up2 = nn.ConvTranspose2d(
            base_channels * 2,
            base_channels,
            kernel_size=4,
            stride=2,
            padding=1
        )

        self.block_up2 = ConvBlock(
            base_channels * 2,
            base_channels,
            time_dim
        )

        self.out = nn.Conv2d(
            base_channels,
            image_channels,
            kernel_size=1
        )

    def forward(self, x, t, labels=None):
        # 时间编码
        emb = self.time_mlp(t)

        # 条件扩散：加入类别标签信息
        if self.conditional:
            if labels is None:
                raise ValueError("Conditional model needs labels.")
            label_emb = self.label_embedding(labels)
            emb = emb + label_emb

        # Encoder
        x0 = self.inc(x, emb)          # [B, 64, 32, 32]

        x1 = self.down1(x0)           # [B, 128, 16, 16]
        x1 = self.block1(x1, emb)

        x2 = self.down2(x1)           # [B, 256, 8, 8]
        x2 = self.block2(x2, emb)

        # Bottleneck
        b = self.bot(x2, emb)         # [B, 256, 8, 8]

        # Decoder
        u1 = self.up1(b)              # [B, 128, 16, 16]
        u1 = torch.cat([u1, x1], dim=1)
        u1 = self.block_up1(u1, emb)

        u2 = self.up2(u1)             # [B, 64, 32, 32]
        u2 = torch.cat([u2, x0], dim=1)
        u2 = self.block_up2(u2, emb)

        predicted_noise = self.out(u2)

        return predicted_noise


if __name__ == "__main__":
    batch_size = 8
    image_size = 32
    num_classes = 10

    x = torch.randn(batch_size, 3, image_size, image_size)
    t = torch.randint(0, 300, (batch_size,))
    labels = torch.randint(0, num_classes, (batch_size,))

    # 条件模型测试
    model = DenoiseUNet(
        image_channels=3,
        num_classes=num_classes,
        conditional=True
    )

    out = model(x, t, labels)

    print("条件扩散模型测试")
    print("输入图片形状:", x.shape)
    print("时间步形状:", t.shape)
    print("标签形状:", labels.shape)
    print("输出噪声形状:", out.shape)

    # 无条件模型测试
    model_uncond = DenoiseUNet(
        image_channels=3,
        num_classes=num_classes,
        conditional=False
    )

    out_uncond = model_uncond(x, t)

    print("无条件扩散模型测试")
    print("输出噪声形状:", out_uncond.shape)