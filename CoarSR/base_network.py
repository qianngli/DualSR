import torch
import torch.nn as nn
import math
import torch.nn.functional as F

def default_conv2d(in_channels, out_channels, kernel_size=3, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias)

def default_conv3d(in_channels, out_channels,  kernel_size=3, stride=1, padding=1, bias=True):
    return nn.Conv3d(
        in_channels, out_channels, kernel_size, stride, 
        padding, bias=bias)        

class ResBlock(nn.Module):
    def __init__(self, conv2d, wn, n_feats, kernel_size=3, bias=True, bn=False, act=nn.ReLU(inplace=True)):

        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(wn(conv2d(n_feats, n_feats, kernel_size, bias=bias)))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if i == 0:
                m.append(act)

        self.body = nn.Sequential(*m)

    def forward(self, x):
        res = x
        x = self.body(x)
        x = torch.add(x, res)
        return x   

class threeUnit(nn.Module):
    def __init__(self, conv3d, wn, n_feats, bias=True, bn=False, act=nn.ReLU(inplace=True)):
        super(threeUnit, self).__init__()    

        self.spatial = wn(conv3d(n_feats, n_feats, kernel_size=(1,3,3), padding=(0,1,1), bias=bias))        
        self.spectral = wn(conv3d(n_feats, n_feats, kernel_size=(3,1,1), padding=(1,0,0), bias=bias))

        self.spatial_one = wn(conv3d(n_feats, n_feats, kernel_size=(1,3,3), padding=(0,1,1), bias=bias))                
        self.relu = act
                 
    def forward(self, x):

        out = self.spatial(x) + self.spectral(x)
        out = self.relu(out)
        out = self.spatial_one(out)
                                
        return out                                   
                     
        
class Upsampler(nn.Sequential):
    def __init__(self, conv2d, wn, scale, n_feats, act=False, bias=True):

        m = []
        if (scale & (scale - 1)) == 0:    # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(wn(conv2d(n_feats, 4 * n_feats, 3, bias)))
                m.append(nn.PixelShuffle(2))

                if act == 'relu':
                    m.append(nn.ReLU(inplace=True))

        elif scale == 3:
            m.append(wn(conv2d(n_feats, 9 * n_feats, 3, bias)))
            m.append(nn.PixelShuffle(3))

            if act == 'relu':
                m.append(nn.ReLU(inplace=True))
        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*m)                      