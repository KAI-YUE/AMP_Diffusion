import functools

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm

from deeplearning.nets import layers

class Generator(nn.Module):
    def __init__(self, in_channels=2, ResBlocks=1, **kwargs):
        """
        in_channels:        original image( rgb|gray ) + edge (binary)
        ResBlocks:          number of residual blocks in the middle part
        imgTrans:           indicate whether or not input image has been normalized
        """
        super(Generator, self).__init__()
        self.in_channels = in_channels

        # Encoder part
        self.ReflectPad = nn.ReflectionPad2d(2)
        self.Conv1 = ConvBlock(self.in_channels, 64, 5, stride=1, padding=0)
        self.Conv2 = ConvBlock(64, 128, 3, stride=2, padding=1)
        self.Conv3 = ConvBlock(128, 256, 3, stride=2, padding=1)
        self.Conv4 = ConvBlock(256, 512, 3, stride=2, padding=1)

        # Middle resnet block
        # blocks = []
        # for i in range(ResBlocks):
        #     blocks.append(ResnetBlock(512, 2))

        # self.mid = nn.Sequential(*blocks)
        self.mid = nn.MaxPool2d(kernel_size=2, stride=2)

        # Decoder part
        self.ConvT1 = ConvTransBlock(512, 512, 4, stride=2, padding=1)
        self.Conv5 = ConvBlock(512, 512, 3, stride=1, padding=1)

        self.ConvT2 = ConvTransBlock(768, 768, 4, stride=2, padding=1)
        self.Conv6 = ConvBlock(768, 256, 3, stride=1, padding=1)

        self.ConvT3 = ConvTransBlock(384, 384, 4, stride=2, padding=1)
        self.Conv7 = ConvBlock(384, 128, 3, stride=1, padding=1)

        self.Conv8 = nn.Conv2d(128, 64, 3, stride=1, padding=1)
        self.Conv9= nn.Conv2d(64, 3, 3, stride=1, padding=1)
        

    def forward(self, x, *args):
        # Encoder part
        x1 = self.Conv1(self.ReflectPad(x))
        x2 = self.Conv2(x1)
        x3 = self.Conv3(x2)
        x4 = self.Conv4(x3)
        
        # Resisual Net block
        # xd = self.mid(x4)
        
        # Decode part
        xd = self.ConvT1(x4)
        xd = self.Conv5(xd)

        xd = self.ConvT2(torch.cat((xd, x3), dim=1))
        xd = self.Conv6(xd)

        xd = self.ConvT3(torch.cat((xd, x2), dim=1))
        xd = self.Conv7(xd)

        xd = self.Conv8(xd)
        xd = self.Conv9(xd)

        # xd = torch.tanh(xd)
        xd = (torch.tanh(xd) + 1) / 2

        return xd
     
class Ccgenerator(nn.Module):
    def __init__(self, in_channels=3, bedding_size=420, **kwargs):
        """
        in_channels:        original image( rgb|gray ) + edge (binary)
        ResBlocks:          number of residual blocks in the middle part
        imgTrans:           indicate whether or not input image has been normalized
        """
        super(Ccgenerator, self).__init__()
        self.in_channels = in_channels

        # which bn (from bigGAN)
        # self.which_linear = nn.Linear
        # bn_linear = functools.partial(self.which_linear, bias=False)

        self.which_embedding = nn.Embedding
        bn_linear = self.which_embedding

        self.which_bn = functools.partial(layers.ccbn,
                        which_linear=bn_linear,
                        cross_replica=False,
                        mybn=False,
                        input_size=1000,
                        norm_style="bn",
                        eps=1.e-5)

        # Encoder part
        self.ReflectPad = nn.ReflectionPad2d(2)
        self.Conv1 = ConvBlock(self.in_channels, 64, 5, stride=1, padding=0)
        self.Conv2 = ConvBlock(64, 128, 3, stride=2, padding=1)
        self.Conv3 = ConvBlock(128, 256, 3, stride=2, padding=1)
        self.Conv4 = ConvBlock(256, 512, 3, stride=2, padding=1)

        # Middle resnet block
        # blocks = []
        # for i in range(ResBlocks):
        #     blocks.append(ResnetBlock(512, 2))

        # self.mid = nn.Sequential(*blocks)
        self.mid = nn.MaxPool2d(kernel_size=2, stride=2)

        # Decoder part
        self.ConvT1 = ConvTransBlock(512, 512, 4, stride=2, padding=1, bn=self.which_bn)
        self.Conv5 = ConvBlock(512, 512, 3, stride=1, padding=1, bn=self.which_bn)

        self.ConvT2 = ConvTransBlock(768, 768, 4, stride=2, padding=1, bn=self.which_bn)
        self.Conv6 = ConvBlock(768, 256, 3, stride=1, padding=1, bn=self.which_bn)

        self.ConvT3 = ConvTransBlock(384, 384, 4, stride=2, padding=1, bn=self.which_bn)
        self.Conv7 = ConvBlock(384, 128, 3, stride=1, padding=1, bn=self.which_bn)

        self.Conv8 = nn.Conv2d(192, 64, 3, stride=1, padding=1)
        self.Conv9= nn.Conv2d(self.in_channels+64, in_channels, 3, stride=1, padding=1)
        
   
    def forward(self, x, y):
        # Encoder part
        x1 = self.Conv1(self.ReflectPad(x))
        x2 = self.Conv2(x1)
        x3 = self.Conv3(x2)
        x4 = self.Conv4(x3)

        # Decode part
        xd = self.ConvT1(x4, y)
        xd = self.Conv5(xd, y)

        xd = self.ConvT2(torch.cat((xd, x3), dim=1), y)
        xd = self.Conv6(xd, y)

        xd = self.ConvT3(torch.cat((xd, x2), dim=1), y)
        xd = self.Conv7(xd, y)
        
        xd = self.Conv8(torch.cat((xd, x1), dim=1))

        xd = self.Conv9(torch.cat((xd, x), dim=1))

        # xd = torch.tanh(xd)
        xd = (torch.tanh(xd) + 1) / 2

        return xd
     

class Discriminator(nn.Module):
    def __init__(self, in_channels, UseInstanceNorm=False, UseSpectralNorm=True, ResBlocks=0):
        super(Discriminator, self).__init__()

        self.in_channels = in_channels
        self.UseInstanceNorm = UseInstanceNorm
        self.UseSpectralNorm = UseSpectralNorm

        if self.UseInstanceNorm:
            self.part1 = nn.Sequential(
                self.spectral_norm(nn.Conv2d(self.in_channels, 64, 3, stride=2, padding=1)),
                nn.InstanceNorm2d(64),
                nn.LeakyReLU(0.2, inplace=True),

                self.spectral_norm(nn.Conv2d(64, 128, 3, stride=2, padding=1)),
                nn.InstanceNorm2d(128),
                nn.LeakyReLU(0.2, inplace=True),

                self.spectral_norm(nn.Conv2d(128, 256, 3, stride=1, padding=1)),
                nn.InstanceNorm2d(256),
                nn.LeakyReLU(0.2, inplace=True)
            )
        else:
            self.part1 = nn.Sequential(
                self.spectral_norm(nn.Conv2d(self.in_channels, 64, 3, stride=2, padding=1)),
                nn.BatchNorm2d(64),
                nn.LeakyReLU(0.2, inplace=True),

                self.spectral_norm(nn.Conv2d(64, 128, 3, stride=2, padding=1)),
                nn.BatchNorm2d(128),
                nn.LeakyReLU(0.2, inplace=True),

                self.spectral_norm(nn.Conv2d(128, 256, 3, stride=1, padding=1)),
                nn.BatchNorm2d(256),
                nn.LeakyReLU(0.2, inplace=True),

                # self.spectral_norm(nn.Conv2d(256, 512, 3, stride=1, padding=1)),
                # nn.BatchNorm2d(512),
                # nn.LeakyReLU(0.2, inplace=True),
            )

        blocks = []
        for i in range(ResBlocks):
            blocks.append(ResnetBlock(512, 1, 'leaky_relu'))

        blocks.append(self.spectral_norm(nn.Conv2d(256, 1, 3, stride=1, padding=1)))
        blocks.append(nn.LeakyReLU(0.2, inplace=True))

        self.part2 = nn.Sequential(*blocks)

    def forward(self, x):
        x = self.part1(x)
        x = self.part2(x)
        return torch.sigmoid(x)

    def spectral_norm(self, x):
        if (self.UseSpectralNorm):
            return spectral_norm(x)
        else:
            return x


class ResnetBlock(nn.Module):
    def __init__(self, dim, dilation=1, activate='relu'):
        super(ResnetBlock, self).__init__()

        self.activate = activate
        self.basic_block = nn.Sequential(
            nn.ReflectionPad2d(dilation),
            nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, padding=0, dilation=dilation),
            nn.InstanceNorm2d(dim),
            self.relu(),

            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, padding=0, dilation=1),
            nn.InstanceNorm2d(dim),
        )

    def forward(self, x):
        return x + self.basic_block(x)
    
    def relu(self):
        if (self.activate == 'relu'):
            return nn.ReLU(inplace=True)
        elif (self.activate == 'leaky_relu'):
            return nn.LeakyReLU(0.2, inplace=True)

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernal, stride, padding, bn=None):
        super(ConvBlock, self).__init__()
        
        # self.conv_block = nn.Sequential(
        #     nn.Conv2d(in_channels, out_channels, kernal, stride, padding),
        #     nn.InstanceNorm2d(out_channels),
        #     nn.ReLU(inplace=True)
        #     # nn.SiLU(inplace=True)
        # )

        if bn == None:
            bn = nn.InstanceNorm2d

        self.use_instance_norm = False
        if isinstance(bn, nn.InstanceNorm2d):
            self.use_instance_norm = True
        
        self.conv = nn.Conv2d(in_channels, out_channels, kernal, stride, padding)
        self.bn = bn(out_channels)
        self.activation = nn.ReLU(inplace=True)
    
    def forward(self, x, y=None):
        if self.use_instance_norm or y is None:
            x = self.bn(self.conv(x))
        else:
            x = self.bn(self.conv(x), y)
        x = self.activation(x)
        return x

class ConvTransBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernal, stride, padding, bn=None):
        super(ConvTransBlock, self).__init__()

        if bn == None:
            self.bn = nn.InstanceNorm2d(out_channels)
        else:
            self.bn = bn(out_channels)

        self.use_instance_norm = False
        if isinstance(bn, nn.InstanceNorm2d):
            self.use_instance_norm = True

        self.convT = nn.ConvTranspose2d(in_channels, out_channels, kernal, stride, padding)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x, y=None):
        # return self.convT_block(x)

        if self.use_instance_norm or y is None:
            out = self.bn(x)
        else:
            out = self.bn(x, y)
            

        out = self.convT(x)
        out = self.activation(out)

        return out
    

def init_weights(module, init_type='normal', gain=0.02):
    '''
    initialize network's weights
    init_type: normal | xavier | kaiming | orthogonal
    '''
    classname = module.__class__.__name__
    if hasattr(module, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
        if init_type == 'normal':
            nn.init.normal_(module.weight.data, 0.0, gain)
        elif init_type == 'xavier':
            nn.init.xavier_normal_(module.weight.data, gain=gain)
        elif init_type == 'kaiming':
            nn.init.kaiming_normal_(module.weight.data, a=0, mode='fan_in')
        elif init_type == 'orthogonal':
            nn.init.orthogonal_(module.weight.data, gain=gain)

        if hasattr(module, 'bias') and module.bias is not None:
            nn.init.constant_(module.bias.data, 0.0)

    elif classname.find('BatchNorm2d') != -1:
        nn.init.normal_(module.weight.data, 1.0, gain)
        nn.init.constant_(module.bias.data, 0.0)