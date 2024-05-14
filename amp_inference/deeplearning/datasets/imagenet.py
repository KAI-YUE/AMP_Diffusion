import os
import pickle
import numpy as np
from PIL import Image

import torch
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import InterpolationMode
from torch import tensor, long

def ImageNet(data_path, config, **kwargs):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    channel = 3
    # im_size = (64, 64)
    im_size = (128, 128)
    num_classes = 10

    transform = T.Compose([
        # T.Resize(config.sample_size),
        T.RandomResizedCrop(im_size[0], scale=(0.6, 1.)),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        # T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    with open(os.path.join(data_path, "train.dat"), "rb") as fp:
        dataset = pickle.load(fp)

    datapair = dataset["data_pair"]
    root_dir = dataset["root"]

    dataset = ImageDataset(root_dir, datapair, transform)

    properties = {
        "channel": channel,
        "im_size": im_size,
        "num_classes": num_classes,
        "n_train": len(dataset),
        "dst_train": dataset,
        "dst_test": dataset,
        "ram_load": False,
        "mean": mean,
        "std": std,
        "signal_type": "rgb",
    }

    class dataset_properties: pass
    for key, value in properties.items():
        setattr(dataset_properties, key, value)

    return dataset_properties


class ImageDataset(Dataset):
    def __init__(self, root_dir, datapair, transform):
        self.root_dir = root_dir
        self.datapair = datapair
        self.transform = transform

    def __getitem__(self, index):
        
        img_name = os.path.join(self.root_dir, self.datapair[index][0], self.datapair[index][1])
        # img = Image.open(img_name).convert('RGB')
        img = Image.open(img_name).convert('RGB')

        img = self.transform(img)
        return img, tensor(self.datapair[index][2], dtype=long)

    def __len__(self):
        return len(self.datapair)

# def fetch_dataloader(config, shuffled=True):
#     """Loads dataset and returns corresponding data loader."""

#     if "imagenet" in config.train_data_dir:
#         transform = T.Compose([
#             # T.Resize(config.sample_size),
#             T.RandomResizedCrop(64, scale=(0.6, 1.)),
#             T.RandomHorizontalFlip(),
#             T.ToTensor(),
#             # T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
#         ])

#         with open(os.path.join(config.train_data_dir, "datapair.dat"), "rb") as fp:
#             record = pickle.load(fp)

#         datapair = record["data_pair"]
#         root_dir = record["root"]
#         dataset = NoisyImageDataset(root_dir, datapair, transform, config.noise_param)
        
#         num_train = int(0.8*len(dataset))
#         num_val = len(dataset) - num_train
#         train_set, val_set = torch.utils.data.random_split(dataset, [num_train, num_val])
        
#         train_loader = DataLoader(dataset=train_set, batch_size=config.batch_size, shuffle=True, num_workers=8, drop_last=True)
#         val_loader = DataLoader(dataset=val_set, batch_size=config.batch_size, shuffle=False, num_workers=8, drop_last=True)
#         return train_loader, val_loader
    

def fetch_paireddataloader(config):
    """Loads dataset and returns corresponding data loader."""

    transform = T.Compose([
        T.Resize(config.sample_size),
        T.ToTensor(),
    ])

    with open(os.path.join(config.train_data_dir, "datapair.dat"), "rb") as fp:
        record = pickle.load(fp)

    datapair = record["data_pair"]
    root_dir = record["root"]
    dataset = PairedCorruptedDataset(root_dir, datapair, transform)
    
    num_train = int(0.9*len(dataset))
    num_val = len(dataset) - num_train
    train_set, val_set = torch.utils.data.random_split(dataset, [num_train, num_val])
    
    train_loader = DataLoader(dataset=train_set, batch_size=config.batch_size, shuffle=True, num_workers=8, drop_last=True)
    val_loader = DataLoader(dataset=val_set, batch_size=config.batch_size, shuffle=False, num_workers=8, drop_last=True)
    return train_loader, val_loader




#===================================================================================================
# noisy image dataset that only adds Gaussian noise
class NoisyImageDataset(Dataset):
    def __init__(self, root_dir, datapair, transform, noise_param):
        """Initialize and preprocess the image dataset."""
        self.root_dir = root_dir
        self.datapair = datapair
        self.transform = transform
        self.num_images = len(self.datapair)
        self.noise_param = noise_param
        self.offset = 0.

        self.std_bound = 100
        self.denominator = 255

    def _add_noise(self, img):
        """Adds Gaussian or Poisson noise to image."""
        std = np.random.uniform(0, self.noise_param)
        noise = torch.randn_like(img) * std

        return img + noise

    def __len__(self):
        """Return the number of images."""
        return self.num_images
    
    def create_iterator(self, batch_size):
        while True:
            sample_loader = DataLoader(
                dataset=self,
                batch_size=batch_size,
                drop_last=True
            )

            for item in sample_loader:
                yield item


    def __getitem__(self, index):
        """Return one image and its corresponding attribute label."""
        img_name = os.path.join(self.root_dir, self.datapair[index][0], 
                       self.datapair[index][1])
        img = Image.open(img_name).convert('RGB') 
        
        image = self.transform(img)
        label = self.datapair[index][2]

        # generate a random noise 
        std_real = np.random.uniform(0, self.std_bound)
        noise = torch.randn_like(image) * std_real/self.denominator

        image_corrput = image + noise
        std = torch.tensor(std_real, dtype=torch.int)
        
        return (image, image_corrput, std, std_real, label)


#===================================================================================================
# corrupt image dataset that is used in USENIX paper
class CorruptedImageDataset(Dataset):
    """Dataset class for the image dataset."""

    def __init__(self, root_dir, datapair, transform, noise_param):
        """Initialize and preprocess the image dataset."""
        self.root_dir = root_dir
        self.datapair = datapair
        self.transform = transform
        self.num_images = len(self.datapair)
        self.noise_param = noise_param
        self.offset = 0.

    def __getitem__(self, index):
        """Return one image and its corresponding attribute label."""
        img_name = os.path.join(self.root_dir, self.datapair[index][0], 
                       self.datapair[index][1])
        img = Image.open(img_name).convert('RGB') 
        
        image_tensor = self.transform(img)
        
        # downsample the image for the super resolver
        interp_modes = [InterpolationMode.NEAREST, InterpolationMode.BILINEAR, InterpolationMode.BICUBIC] 
        idx = np.random.randint(0, len(interp_modes))
        interp_mode = interp_modes[idx]

        gaussian_kernel = np.random.randint(0, 2)

        if gaussian_kernel:
            lr_transform = T.Compose([
                T.Resize(size=(32,32), interpolation=interp_mode),  
                # T.ColorJitter(brightness=.2, hue=.1), 
                # T.Resize(size=(128,128), interpolation=interp_mode),
                T.Resize(size=(64,64), interpolation=interp_mode),
                T.GaussianBlur(kernel_size=(3, 3), sigma=(0.1,1))])
        else:
            lr_transform = T.Compose([
                T.Resize(size=(32,32), interpolation=interp_mode), 
                # T.Resize(size=(128,128), interpolation=interp_mode)
                T.Resize(size=(64,64), interpolation=interp_mode),
                ])

        noise_first = np.random.randint(0, 2)
        if noise_first:
            image_corrput = self._add_noise(image_tensor)
            image_corrput = lr_transform(image_corrput)
        else:
            image_corrput = lr_transform(image_tensor)
            image_corrput = self._add_noise(image_corrput)

        return image_corrput, image_tensor, self.datapair[index][2]
        
    def _add_noise(self, img):
        """Adds Gaussian or Poisson noise to image."""
        std = np.random.uniform(0, self.noise_param)
        noise = torch.randn_like(img) * std

        return img + noise

    def __len__(self):
        """Return the number of images."""
        return self.num_images
    
    def create_iterator(self, batch_size):
        while True:
            sample_loader = DataLoader(
                dataset=self,
                batch_size=batch_size,
                drop_last=True
            )

            for item in sample_loader:
                yield item


class PairedCorruptedDataset(Dataset):
    """Dataset class for the image dataset."""

    def __init__(self, root_dir, datapair, transform):
        """Initialize and preprocess the image dataset."""
        self.root_dir = root_dir
        self.ori_root_dir = root_dir.replace("synthimage", "image")
        self.datapair = datapair
        self.transform = transform
        self.num_images = len(self.datapair)

    def __getitem__(self, index):
        """Return one image and its corresponding attribute label."""
        img_name = os.path.join(self.root_dir, self.datapair[index][0], 
                       self.datapair[index][1])

        label = self.datapair[index][2]

        img = Image.open(img_name).convert('RGB') 
        image_corrput = self.transform(img)
        
        ori_img_name = os.path.join(self.ori_root_dir, self.datapair[index][0], self.datapair[index][1])
        ori_img = Image.open(ori_img_name).convert('RGB')
        image_tensor = self.transform(ori_img)

        cat_imgs = torch.cat([image_corrput, image_tensor], dim=0)
        aug_transform = T.Compose([
            T.RandomHorizontalFlip()])

        aug_images = aug_transform(cat_imgs)
        image_corrput = aug_images[:3]
        image_tensor = aug_images[3:]
        
        return image_corrput, image_tensor, label
        
    def __len__(self):
        """Return the number of images."""
        return self.num_images
    
    def create_iterator(self, batch_size):
        while True:
            sample_loader = DataLoader(
                dataset=self,
                batch_size=batch_size,
                drop_last=True
            )

            for item in sample_loader:
                yield item


