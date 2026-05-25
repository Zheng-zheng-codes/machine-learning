import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def get_mnist_loader(
    data_root="../data",
    batch_size=128,
    num_workers=2,
    train=True
):

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.5,),
            std=(0.5,)
        )
    ])

    dataset = datasets.MNIST(
        root=data_root,
        train=train,
        download=False,
        transform=transform
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True
    )

    return loader


if __name__ == "__main__":
    train_loader = get_mnist_loader()

    images, labels = next(iter(train_loader))

    print("MNIST 数据集读取成功")
    print("一批图片的形状:", images.shape)
    print("一批标签的形状:", labels.shape)
    print("图片像素范围:", images.min().item(), images.max().item())