import torch
import torch.nn as nn

def get_norm(norm_type, channels):
    if norm_type == "bn":
        return nn.BatchNorm2d(channels)
    elif norm_type == "ln":
        return nn.GroupNorm(1, channels)
    else:
        raise ValueError("norm_type 只能是 bn 或 ln")

#模型构建
class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride = 1, norm_type="bn"):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride = stride, padding = 1, bias = False)
        self.bn1 = get_norm(norm_type, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size = 3, stride = 1, padding = 1, bias = False)
        self.bn2 = get_norm(norm_type, out_channels)
        if in_channels != out_channels or stride != 1:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                get_norm(norm_type, out_channels)
            )
        else:
            self.shortcut = nn.Identity()
    
    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = torch.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = out + self.shortcut(x)
        out = torch.relu(out)
        return out

#模型的入口
class ResNet18(nn.Module):
    def __init__(self, num_classes = 10, norm_type="bn"):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size = 3, stride = 1, padding = 1, bias = False)
        self.bn1 = get_norm(norm_type, 64)
        self.layer1 = nn.Sequential(
            BasicBlock(64, 64, norm_type=norm_type),
            BasicBlock(64, 64, norm_type=norm_type)
        )
        self.layer2 = nn.Sequential(
            BasicBlock(64, 128, stride = 2, norm_type=norm_type),
            BasicBlock(128, 128, norm_type=norm_type)
        )
        self.layer3 = nn.Sequential(
            BasicBlock(128, 256, stride = 2, norm_type=norm_type),
            BasicBlock(256, 256, norm_type=norm_type)
        )
        self.layer4 = nn.Sequential(
            BasicBlock(256, 512, stride = 2, norm_type=norm_type),
            BasicBlock(512, 512, norm_type=norm_type)
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = torch.relu(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.pool(out)
        out = out.view(out.size(0),-1)
        out = self.fc(out)
        return out