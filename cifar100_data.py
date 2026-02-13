
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader

def load_cifar100_data(config):
    #Define transformations for the training set
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)), # CIFAR100 mean and std
    ])
    #Define transformations for the validation set
    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])
    #Load CIFAR100
    train_dataset = datasets.CIFAR100(root=r'C:\Users\guttormsonic\machinelearning\Assignment1\CIFAR100.ds', train=True, download=True, transform=transform_train)
    val_dataset = datasets.CIFAR100(root=r'C:\Users\guttormsonic\machinelearning\Assignment1\CIFAR100.ds', train=False, download=True, transform=transform_val)
    #Data Loaders
    batch_size = config['batch_size']
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    input_channels = 3  #3 colors RGB
    image_size = 32     #CIFAR100 is 32x32
    output_classes = 100 # 100 classes for CIFAR100
    print(f"CIFAR100 - Input Channels: {input_channels}, Image Size: {image_size}x{image_size}, Output Classes: {output_classes}")
    print(f"Number of training samples: {len(train_dataset)}")
    print(f"Number of validation samples: {len(val_dataset)}")
    return train_loader, val_loader, input_channels, image_size, output_classes
