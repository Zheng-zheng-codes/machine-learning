import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

transform = transforms.Compose([
    transforms.ToTensor()
])

def get_dataset(name, train=True):
    if name == "cifar10":
        cifar10data = torchvision.datasets.CIFAR10(
            root = './data',
            train = train,
            download = True,
            transform = transform
        )
        return cifar10data

    elif name == "cifar100":
        cifar100data = torchvision.datasets.CIFAR100(
            root='./data',
            train=train,
            download=True,
            transform=transform
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