import os
import torch
import torch.nn as nn
import torchvision.models as models
from collections import namedtuple

import torchvision

class AdvLoss(nn.Module):

    def __init__(self, type='BCE', Device=True):
        r"""
        type = BCE | MSE 
        """
        super(AdvLoss, self).__init__()

        device = torch.device('cuda' if Device else 'cpu')
        self.type = type
        self.real_label = torch.tensor(1.).to(device)
        self.fake_label = torch.tensor(0.).to(device)

        if type == 'BCE':
            self.criterion = nn.BCELoss()

        elif type == 'MSE':
            self.criterion = nn.MSELoss()


    def __call__(self, outputs, is_real):

        labels = (self.real_label if is_real else self.fake_label).expand_as(outputs)
        loss = self.criterion(outputs, labels)
        return loss

class vgg_model(nn.Module):
    def __init__(self, type='vgg16'):
        super(vgg_model, self).__init__()
        
        self.type = type

        if (type == 'vgg19'):
            self.model = models.vgg19(pretrained=True).eval()
        elif (type == 'vgg16'):
            self.model = models.vgg16(pretrained=True).eval()
        
        for p in self.model.parameters():
            p.requires_grad = False


# --------------------------------------------
# Perceptual loss
# --------------------------------------------
class VGGFeatureExtractor(nn.Module):
    def __init__(self, feature_layer=[2,7,16,25,34], use_input_norm=True, use_range_norm=False):
        super(VGGFeatureExtractor, self).__init__()
        '''
        use_input_norm: If True, x: [0, 1] --> (x - mean) / std
        use_range_norm: If True, x: [0, 1] --> x: [-1, 1]
        '''
        model = torchvision.models.vgg19(pretrained=True)
        model = model.cuda()
        self.use_input_norm = use_input_norm
        self.use_range_norm = use_range_norm
        if self.use_input_norm:
            mean = torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
            std = torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
            self.register_buffer('mean', mean)
            self.register_buffer('std', std)
        self.list_outputs = isinstance(feature_layer, list)
        if self.list_outputs:
            self.features = nn.Sequential()
            feature_layer = [-1] + feature_layer
            for i in range(len(feature_layer)-1):
                self.features.add_module('child'+str(i), nn.Sequential(*list(model.features.children())[(feature_layer[i]+1):(feature_layer[i+1]+1)]))
        else:
            self.features = nn.Sequential(*list(model.features.children())[:(feature_layer + 1)])

        # print(self.features)

        # No need to BP to variable
        for k, v in self.features.named_parameters():
            v.requires_grad = False

    def forward(self, x):
        if self.use_range_norm:
            x = (x + 1.0) / 2.0
        if self.use_input_norm:
            x = (x - self.mean) / self.std
        if self.list_outputs:
            output = []
            for child_model in self.features.children():
                x = child_model(x)
                output.append(x.clone())
            return output
        else:
            return self.features(x)


class PerceptualLoss(nn.Module):
    """VGG Perceptual loss
    """

    def __init__(self, feature_layer=[2,7,16,25,34], weights=[0.1,0.1,1.0,1.0,1.0], lossfn_type='l1', use_input_norm=False, use_range_norm=True):
        super(PerceptualLoss, self).__init__()
        self.vgg = VGGFeatureExtractor(feature_layer=feature_layer, use_input_norm=use_input_norm, use_range_norm=use_range_norm)
        self.lossfn_type = lossfn_type
        self.weights = weights
        if self.lossfn_type == 'l1':
            self.lossfn = nn.L1Loss()
        else:
            self.lossfn = nn.MSELoss()
        # print(f'feature_layer: {feature_layer}  with weights: {weights}')

    def forward(self, x, gt):
        """Forward function.
        Args:
            x (Tensor): Input tensor with shape (n, c, h, w).
            gt (Tensor): Ground-truth tensor with shape (n, c, h, w).
        Returns:
            Tensor: Forward results.
        """
        x_vgg, gt_vgg = self.vgg(x), self.vgg(gt.detach())
        loss = 0.0
        if isinstance(x_vgg, list):
            n = len(x_vgg)
            for i in range(n):
                loss += self.weights[i] * self.lossfn(x_vgg[i], gt_vgg[i])
        else:
            loss += self.lossfn(x_vgg, gt_vgg.detach())
        return loss

class PercepLoss(vgg_model):
    def __init__(self, loss_type='L1', Device=True):
        super(PercepLoss, self).__init__()
        
        if Device:
            self.model.cuda()
        if (self.type=='vgg19'):     
            self.layerName = {
                '3': "relu1_2",
                '8': "relu2_2",
                '17': "relu3_4",
                '26': "relu4_4"
            }
            self.lossOutput = namedtuple("LossOutput", ["relu1_2", "relu2_2", "relu3_4", "relu4_4"])
            
        elif (self.type=='vgg16'):
            self.layerName = {
                '3': "relu1_2",
                '8': "relu2_2",
                '15': "relu3_3",
                '22': "relu4_3"
            }
            self.lossOutput = namedtuple("LossOutput", ["relu1_2", "relu2_2", "relu3_3", "relu4_3"])
            
        
        if (loss_type == 'L1'):
            self.loss = nn.L1Loss()
        else:
            self.loss = nn.MSELoss()
            
            
    def __call__(self, x, y):
        loss = 0.

        assert(x.shape == y.shape)
        if (x.shape == 3):
            ch, h, w = x.shape
            b = 1
        else:
            b , ch, h, w = x.shape

        if (ch == 1):
            x = x.expand(b, 3, h, w)
            y = y.expand(b, 3, h, w)
             
        for name, module in self.model.features._modules.items():
            x = module(x)
            y = module(y)
            if name in self.layerName:
                loss += self.loss(x, y)

        return loss


class StyleLoss(vgg_model):
    def __init__(self, loss_type='L1', Device=True):
        super(StyleLoss, self).__init__()
        
        if Device:
            self.model.cuda()
        if (self.type=='vgg19'):     
            self.layerName = {
                '3': "relu1_2",
                '8': "relu2_2",
                '17': "relu3_4",
                '26': "relu4_4"
            }
            self.lossOutput = namedtuple("LossOutput", ["relu1_2", "relu2_2", "relu3_4", "relu4_4"])
            
        elif (self.type=='vgg16'):
            self.layerName = {
                '3': "relu1_2",
                '8': "relu2_2",
                '15': "relu3_3",
                '22': "relu4_3"
            }
            self.lossOutput = namedtuple("LossOutput", ["relu1_2", "relu2_2", "relu3_3", "relu4_3"])
        
        if (loss_type == 'L1'):
            self.loss = nn.L1Loss()
        else:
            self.loss = nn.MSELoss()
        
    def __call__(self, x, y):
        
        loss = 0.

        assert(x.shape == y.shape)
        if (x.shape == 3):
            ch, h, w = x.shape
            b = 1
        else:
            b , ch, h, w = x.shape

        if (ch == 1):
            x = x.expand(b, 3, h, w)
            y = y.expand(b, 3, h, w)

        for name, module in self.model.features._modules.items():
            x = module(x)
            y = module(y)
            if name in self.layerName:
                loss += self.loss(self.gram(x), self.gram(y))

        return loss
        
    def gram(self, x):
        
        if (len(x.shape) == 3):
            ch, h, w = x.shape
            b = 1
        else:
            b, ch, h, w = x.shape
            
        f = x.view(b, ch, w * h)
        f_T = f.transpose(1, 2)
        G = f.bmm(f_T) / (h * w * ch)
        return G


            
            
        
        
    
    

