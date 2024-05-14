import os
# PyTorch Libraries
import torch

# My Libraries
from deeplearning.nets.resnet import *
from deeplearning.nets.preresnet import *
from deeplearning.nets.lenet import LeNet, LeNet_5
from deeplearning.nets.vgg import VGG_7
from deeplearning.nets.unet import UNet
from deeplearning.nets.ccunet import Ccgenerator, Generator

def init_weights(module, init_type='normal', gain=0.01):
    '''
    initialize network's weights
    init_type: normal | uniform | kaiming  
    '''
    classname = module.__class__.__name__
    if hasattr(module, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
        if init_type == 'normal':
            nn.init.normal_(module.weight.data, 0.0, gain)
        elif init_type == "uniform":
            nn.init.uniform_(module.weight.data)
        elif init_type == 'kaiming':
            nn.init.kaiming_normal_(module.weight.data, a=0, mode='fan_in')

        if hasattr(module, 'bias') and module.bias is not None:
            nn.init.constant_(module.bias.data, 0.0)

    elif (classname.find('BatchNorm') != -1 and module.weight is not None):
        nn.init.normal_(module.weight.data, 1.0, gain)
        nn.init.constant_(module.bias.data, 0.0)

    elif (classname.find("GroupNorm") != -1 and module.weight is not None):
        nn.init.normal_(module.weight.data, 1.0, gain)
        nn.init.constant_(module.bias.data, 0.0)

nn_registry = {
    "unet":             UNet,
    "cgen":             Ccgenerator,
    "gen":              Generator,

    "resnet18":         ResNet18,
    "resnet34":         ResNet34,
    
    "lenet":            LeNet,
    "lenet_5":          LeNet_5,

    "vgg7":             VGG_7,
}
