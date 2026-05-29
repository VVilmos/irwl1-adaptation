import torchvision
import torch
import irwl1.config as config

def fetch_fmnist():
    fmnist = torchvision.datasets.FashionMNIST(root=f"./data", train=True,
                                                transform=torchvision.transforms.Compose([
                                                    torchvision.transforms.ToTensor(),
                                                    torchvision.transforms.Normalize(mean=(0.5,), std=(0.5,))]), download=False)

    fmnist_train, fmnist_val = torch.utils.data.random_split(fmnist, [50000, 10000])

    fmnist_test = torchvision.datasets.FashionMNIST(root=f"./data", train=False,
                                                    transform=torchvision.transforms.Compose([
                                                        torchvision.transforms.ToTensor(),
                                                        torchvision.transforms.Normalize(mean=(0.5,), std=(0.5,))]),
                                                    download=False)

    train_size, test_size, val_size = len(fmnist_train), len(fmnist_test), len(fmnist_val)

    train_loader = torch.utils.data.DataLoader(fmnist_train, batch_size=train_size, shuffle=True, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(fmnist_val, batch_size=val_size, shuffle=False, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(fmnist_test, batch_size=test_size, shuffle=False, pin_memory=True)


    return train_loader, val_loader, test_loader

def fetch_cifar10():
    cifar10 = torchvision.datasets.CIFAR10(root=f"./data", train=True,
                                           transform=torchvision.transforms.Compose([
                                               torchvision.transforms.ToTensor(),
                                               torchvision.transforms.Normalize(mean=(0.5, 0.5, 0.5),
                                                                                std=(0.5, 0.5, 0.5))]),
                                           download=True)

    cifar10_train, cifar10_val = torch.utils.data.random_split(cifar10, [40000, 10000])

    cifar10_test = torchvision.datasets.CIFAR10(root=f"./data", train=False,
                                                transform=torchvision.transforms.Compose([
                                                    torchvision.transforms.ToTensor(),
                                                    torchvision.transforms.Normalize(mean=(0.5, 0.5, 0.5),
                                                                                     std=(0.5, 0.5, 0.5))]),
                                                download=True)

    train_size, test_size, val_size = len(cifar10_train), len(cifar10_test), len(cifar10_val)

    train_loader = torch.utils.data.DataLoader(cifar10_train, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(cifar10_val, batch_size=config.VAL_BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(cifar10_test, batch_size=config.VAL_BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    return train_loader, val_loader, test_loader

def fetch_cifar10_test_mini():
    cifar10_test = torchvision.datasets.CIFAR10(root=f"./data", train=False,
                                                transform=torchvision.transforms.Compose([
                                                    torchvision.transforms.ToTensor(),
                                                    torchvision.transforms.Normalize(mean=(0.5, 0.5, 0.5),
                                                                                     std=(0.5, 0.5, 0.5))]),
                                                download=False)

    cifar10_test_mini, _ = torch.utils.data.random_split(cifar10_test, [1000, 9000])

    test_loader = torch.utils.data.DataLoader(cifar10_test_mini, batch_size=config.VAL_BATCH_SIZE, shuffle=False, pin_memory=True)

    return test_loader

def load_to_memory(data_loader):
    image_tensor, label_tensor = next(iter(data_loader)) # one big tensor
    image_tensor = image_tensor.to(config.DEVICE)
    label_tensor = label_tensor.to(config.DEVICE)

    return image_tensor, label_tensor



