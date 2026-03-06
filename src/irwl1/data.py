import torchvision
import torch

def fetch_fmnist():
    fmnist = torchvision.datasets.FashionMNIST(root=f"./data", train=True,
                                                transform=torchvision.transforms.Compose([
                                                    torchvision.transforms.ToTensor(),
                                                    torchvision.transforms.Normalize(mean=(0.5,), std=(0.5,))]), download=True)

    fmnist_train, fmnist_val = torch.utils.data.random_split(fmnist, [50000, 10000])

    fmnist_test = torchvision.datasets.FashionMNIST(root=f"./data", train=False,
                                                    transform=torchvision.transforms.Compose([
                                                        torchvision.transforms.ToTensor(),
                                                        torchvision.transforms.Normalize(mean=(0.5,), std=(0.5,))]),
                                                    download=True)

    return fmnist_train, fmnist_val, fmnist_test
