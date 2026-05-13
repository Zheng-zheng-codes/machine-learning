import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision import datasets

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

# TinyImageNet 数据增强
tiny_train_transform = transforms.Compose([

    transforms.Resize((64, 64)),

    # 64×64随机裁剪
    transforms.RandomCrop(64, padding=4),

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

    # TinyImageNet 标准化
    transforms.Normalize(
        mean=[0.4802, 0.4481, 0.3975],
        std=[0.2302, 0.2265, 0.2262]
    )
])


tiny_test_transform = transforms.Compose([

    transforms.Resize((64, 64)),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.4802, 0.4481, 0.3975],
        std=[0.2302, 0.2265, 0.2262]
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

    elif name == "tinyimagenet":

        if train:
            path = './data/tiny-imagenet-200/train'
        else:
            path = './data/tiny-imagenet-200/val'

        tinydata = datasets.ImageFolder(
            root = path,
            transform = tiny_train_transform if train else tiny_test_transform
        )

        return tinydata

def get_loader(name, train=True, batch_size = 64):
    dataset=get_dataset(name, train)
    loader = DataLoader(
        dataset,
        batch_size = batch_size,
        shuffle = train
    )
    return loader