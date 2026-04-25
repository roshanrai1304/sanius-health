import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms

from . import config

CAT_CLASS = 3
DOG_CLASS = 5


def get_baseline_transforms():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                             std=[0.2470, 0.2435, 0.2616]),
    ])


def get_augmented_transforms():
    return transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2,
                               saturation=0.2, hue=0.1),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                             std=[0.2470, 0.2435, 0.2616]),
    ])


class CatDogDataset(Dataset):
    def __init__(self, cifar_dataset):
        self.data = []
        self.targets = []
        for img, label in zip(cifar_dataset.data, cifar_dataset.targets):
            if label == CAT_CLASS:
                self.data.append(img)
                self.targets.append(0)
            elif label == DOG_CLASS:
                self.data.append(img)
                self.targets.append(1)
        self.data = np.array(self.data)
        self.transform = cifar_dataset.transform

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        from PIL import Image
        img = Image.fromarray(self.data[idx])
        if self.transform:
            img = self.transform(img)
        return img, self.targets[idx]


def create_dataloaders(augment=False):
    train_transform = get_augmented_transforms() if augment else get_baseline_transforms()
    val_transform = get_baseline_transforms()

    train_cifar = datasets.CIFAR10(root=config.DATA_DIR, train=True,
                                   download=True, transform=train_transform)
    val_cifar = datasets.CIFAR10(root=config.DATA_DIR, train=False,
                                  download=True, transform=val_transform)

    train_ds = CatDogDataset(train_cifar)
    val_ds = CatDogDataset(val_cifar)

    train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE,
                              shuffle=True, num_workers=config.NUM_WORKERS)
    val_loader = DataLoader(val_ds, batch_size=config.BATCH_SIZE,
                            shuffle=False, num_workers=config.NUM_WORKERS)

    print(f"Train samples: {len(train_ds)}, Val samples: {len(val_ds)}")
    return train_loader, val_loader
