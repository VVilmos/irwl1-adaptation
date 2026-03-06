import torchvision
import torch
from irwl1.config import BATCH_SIZE, DEVICE

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


    train_loader = torch.utils.data.DataLoader(fmnist_train, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(fmnist_val, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(fmnist_test, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)


    return train_loader, val_loader, test_loader

def load_to_memory(data_loader):
    image_tensor, label_tensor = next(iter(data_loader)) # one big tensor
    image_tensor = image_tensor.to(DEVICE)
    label_tensor = label_tensor.to(DEVICE)

    return image_tensor, label_tensor



