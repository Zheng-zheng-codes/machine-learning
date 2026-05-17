import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# CIFAR 数据增强
cifar_train_transform = transforms.Compose([

    # 随机裁剪
    transforms.RandomCrop(32, padding=4),

    # 随机水平翻转
    transforms.RandomHorizontalFlip(),

    # 随机仿射变换
    transforms.RandomAffine(
        degrees=10,
        translate=(0.1, 0.1),
        scale=(0.9, 1.1),
        shear=10
    ),

    transforms.ToTensor(),

    # CIFAR 标准化
    transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465],
        std=[0.2023, 0.1994, 0.2010]
    )
])


cifar_test_transform = transforms.Compose([

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465],
        std=[0.2023, 0.1994, 0.2010]
    )
])

def get_dataset(name, train=True):
    if name == "cifar10":
        cifar10data = torchvision.datasets.CIFAR10(
            root = './data',
            train = train,
            download = True,
            transform = cifar_train_transform if train else cifar_test_transform
        )
        return cifar10data

    elif name == "cifar100":
        cifar100data = torchvision.datasets.CIFAR100(
            root = './data',
            train = train,
            download = True,
            transform = cifar_train_transform if train else cifar_test_transform
        )
        return cifar100data

def get_loader(name, train=True, batch_size = 64):
    dataset=get_dataset(name, train)
    loader = DataLoader(
        dataset,
        batch_size = batch_size,
        shuffle = train
    )
    return loader