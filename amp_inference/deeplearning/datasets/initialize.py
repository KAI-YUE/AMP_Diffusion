import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader

from deeplearning.datasets.cifar10 import CIFAR10
from deeplearning.datasets.fmnist import FashionMNIST
from deeplearning.datasets.bsd import BSD
from deeplearning.datasets.imagenet import ImageNet

dataset_registry = {
    "cifar10": CIFAR10,
    "fmnist": FashionMNIST,
    
    "bsd": BSD,
    "imagenet": ImageNet,
}


def fetch_dataloader(config, train, test):
    if config.ram_load:
        train_loader = DataLoader(MyDataset(train["images"], train["labels"]), 
                        batch_size=config.batch_size, shuffle=True,
                        num_workers=config.workers, pin_memory=False)
        test_loader = DataLoader(MyDataset(test["images"], test["labels"]), 
                        batch_size=config.batch_size, shuffle=False,
                        num_workers=config.workers, pin_memory=False)

    else:
        train_loader = DataLoader(dataset=train, batch_size=config.batch_size, 
                                    shuffle=True, num_workers=1, drop_last=False)
        test_loader = DataLoader(dataset=test, batch_size=config.batch_size, 
                                    shuffle=True, num_workers=1, drop_last=False)

    return train_loader, test_loader


def fetch_noisydataloader(config, train, test):

    if config.ram_load:
        train_loader = DataLoader(NoisyDataset(train["images"], train["labels"], config=config), 
                        batch_size=config.batch_size, shuffle=True,
                        num_workers=config.workers, pin_memory=False)
        
        test_loader = DataLoader(NoisyDataset(test["images"], test["labels"], config=config), 
                        batch_size=config.batch_size, shuffle=False,
                        num_workers=config.workers, pin_memory=False)
    else:
        train_loader = DataLoader(dataset=train, batch_size=config.batch_size, 
                                  shuffle=True, num_workers=config.workers, drop_last=False)
        test_loader = DataLoader(dataset=test, batch_size=config.batch_size, 
                                  shuffle=True, num_workers=config.workers, drop_last=False)

    return train_loader, test_loader


def fetch_dataset(config):
    dataset = dataset_registry[config.dataset](config.data_path, config=config)

    config.num_classes = dataset.num_classes
    config.im_size = dataset.im_size
    config.channel = dataset.channel
    config.n_train = dataset.n_train
    config.ram_load = dataset.ram_load
    config.data_mean = dataset.mean
    config.data_std = dataset.std

    # signal length (for a grey scale image)
    config.N = config.im_size[0] * config.im_size[1]

    config.signal_type = dataset.signal_type

    return dataset


class MyDataset(Dataset):
    def __init__(self, images, labels):
        """Construct a customized dataset
        """
        if min(labels) < 0:
            labels = (labels).reshape((-1,1)).astype(np.float32)
        else:
            labels = (labels).astype(np.int64)

        self.images = torch.from_numpy(images)
        self.labels = torch.from_numpy(labels)
        self.num_samples = images.shape[0]

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx] 
        return (image, label)


class NoisyDataset(Dataset):
    def __init__(self, images, labels, **kwargs):
        """Construct a customized dataset
        """
        if min(labels) < 0:
            labels = (labels).reshape((-1,1)).astype(np.float32)
        else:
            labels = (labels).astype(np.int64)

        self.images = torch.from_numpy(images)
        self.labels = torch.from_numpy(labels)
        self.num_samples = images.shape[0]

        self.std_bound = kwargs["config"].std_bound
        self.denominator = kwargs["config"].denominator

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        # generate a random noise 
        std_real = np.random.uniform(0, self.std_bound)
        noise = torch.randn_like(image) * std_real/self.denominator

        image_corrput = image + noise

        std = torch.tensor(std_real, dtype=torch.int)
        return (image, image_corrput, std, std_real, label)
