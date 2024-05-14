import os
import pickle
import numpy as np


def CIFAR10(data_path, **kwargs):
    mean = [0.4914, 0.4822, 0.4465]
    std = [0.2470, 0.2435, 0.2616]

    channel = 3
    im_size = (32, 32)
    num_classes = 10

    with open(os.path.join(data_path, "train.dat"), "rb") as fp:
        dst_train = pickle.load(fp)
    
    with open(os.path.join(data_path, "test.dat"), "rb") as fp:
        dst_test = pickle.load(fp)

    # apply normalization
    train_images, test_images = dst_train["images"], dst_test["images"]
    train_images, test_images = train_images.astype(np.float32)/255, test_images.astype(np.float32)/255

    # reshape images    
    train_images = train_images.reshape(-1, 3, 32, 32)
    test_images = test_images.reshape(-1, 3, 32, 32)

    # for i, mean_i in enumerate(mean):
    #     train_images[:, i] = (train_images[:, i] - mean[i])/std[i]
    #     test_images[:, i] = (test_images[:, i] - mean[i])/std[i]

    dst_train["images"], dst_test["images"] = train_images, test_images

    properties = {
        "channel": channel,
        "im_size": im_size,
        "num_classes": num_classes,
        "n_train": len(dst_train["images"]),
        "dst_train": dst_train,
        "dst_test": dst_test,
        "ram_load": True,
        "mean": mean,
        "std": std,
    }

    class dataset_properties: pass
    for key, value in properties.items():
        setattr(dataset_properties, key, value)

    return dataset_properties

