import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from srcs.utils.utils_deblur_dwdn import MedianPool2d
from srcs.utils.utils_deblur_dwdn import get_uperleft_denominator as wiener_deblur_func


# ======
# ref: 
#   https://gitlab.mpi-klsb.mpg.de/jdong/dwdn
#   https://github.com/cszn/KAIR
# ======

# --------------------------------------------
# basic functions
# --------------------------------------------


def pad2same_size(x1, x2):
    '''
    pad x1 or x2 to the same size (the size of the larger one)
    '''
    diffX = x2.size()[3] - x1.size()[3]
    diffY = x2.size()[2] - x1.size()[2]

    if diffX == 0 and diffY == 0:
        return x1, x2

    if diffX >= 0 and diffY >= 0:
        x1 = nn.functional.pad(x1, (diffX // 2, diffX - diffX//2,
                                    diffY // 2, diffY - diffY//2))
    elif diffX < 0 and diffY < 0:
        x2 = nn.functional.pad(
            x2, (-diffX // 2, -diffX + diffX//2, -diffY // 2, -diffY + diffY//2))
    elif diffX >= 0 and diffY < 0:
        x1 = nn.functional.pad(x1, (diffX // 2, diffX - diffX//2, 0, 0))
        x2 = nn.functional.pad(
            x2, (0, 0, -diffY // 2, -diffY + diffY//2))
    elif diffX < 0 and diffY >= 0:
        x1 = nn.functional.pad(x1, (0, 0, diffY // 2, diffY - diffY//2))
        x2 = nn.functional.pad(
            x2, (-diffX // 2, -diffX + diffX//2, 0, 0))

    return x1, x2


def pad2size(x, size):
    '''
    pad x to given size
    x: N*C*H*W
    size: H'*W'
    '''
    diffX = size[1] - x.size()[3]
    diffY = size[0] - x.size()[2]

    if diffX == 0 and diffY == 0:
        return x

    if diffX >= 0 and diffY >= 0:
        x = nn.functional.pad(x, (diffX // 2, diffX - diffX//2,
                                  diffY // 2, diffY - diffY//2))
    elif diffX < 0 and diffY < 0:
        x = x[:, :, -diffY // 2: diffY + (-diffY) //
              2, -diffX // 2: diffX + (-diffX)//2]
    elif diffX >= 0 and diffY < 0:
        x = x[:, :, -diffY // 2: diffY + (-diffY)//2, :]
        x = nn.functional.pad(x, (diffX // 2, diffX - diffX//2, 0, 0))
    elif diffX < 0 and diffY >= 0:
        x = x[:, :, :, -diffX // 2: diffX + (-diffX)//2]
        x = nn.functional.pad(x, (0, 0, diffY // 2, diffY - diffY//2))
    return x


def image_nsr(_input_blur):
    # image noise-signal-ratio (amplitude) N*C
    # ref: DWDN/wiener_filter_para
    median_filter = MedianPool2d(
        kernel_size=3, padding=1)(_input_blur)
    diff = median_filter - _input_blur
    num = (diff.shape[2]*diff.shape[3])
    mean_n = torch.sum(diff, (2, 3), keepdim=True).repeat(
        1, 1, diff.shape[2], diff.shape[3])/num
    var_n2 = (torch.sum((diff - mean_n) * (diff - mean_n),
                        (2, 3))/(num-1))**(0.5)

    var_s2 = (torch.sum((median_filter) * (median_filter), (2, 3))/(num-1))**(0.5)
    # NSR = var_n2 / var_s2 * 8.0 / 3.0 / 10.0
    NSR = var_n2 / var_s2

    return NSR

# --------------------------------------------
# basic modules
# --------------------------------------------


class Conv(nn.Module):
    def __init__(self, input_channels, n_feats, kernel_size, stride=1, padding=0, bias=True, bn=False, act=False):
        super(Conv, self).__init__()
        m = []
        m.append(nn.Conv2d(input_channels, n_feats,
                 kernel_size, stride, padding, bias=bias))
        if bn:
            m.append(nn.BatchNorm2d(n_feats))
        if act:
            m.append(nn.ReLU(True))
        self.body = nn.Sequential(*m)

    def forward(self, input):
        return self.body(input)


class Deconv(nn.Module):
    def __init__(self, input_channels, n_feats, kernel_size, stride=2, padding=0, output_padding=0, bias=True, act=False):
        super(Deconv, self).__init__()
        m = []
        m.append(nn.ConvTranspose2d(input_channels, n_feats, kernel_size,
                 stride=stride, padding=padding, output_padding=output_padding, bias=bias))
        if act:
            m.append(nn.ReLU(True))
        self.body = nn.Sequential(*m)

    def forward(self, input):
        return self.body(input)


class ResBlock(nn.Module):
    def __init__(self, conv, n_feat, kernel_size, padding=0, bias=True, bn=False, act=nn.ReLU(True), res_scale=1):
        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(conv(n_feat, n_feat, kernel_size,
                     padding=padding, bias=bias))
            if bn:
                m.append(nn.BatchNorm2d(n_feat))
            if i == 0:
                m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x

        return res

# --------------------------------------------
# FSWD functions and modules
# --------------------------------------------


def wiener_deblur_mc(x, kernel):
    '''
    multi-channel wiener deblur
    '''
    # normalize kernel
    kernel_norm = torch.zeros_like(kernel).cuda()
    for jj in range(kernel.shape[0]):
        kernel_norm[jj:jj+1, :, :, :] = torch.div(kernel[jj:jj+1, :, :, :], torch.sum(
            kernel[jj:jj+1, :, :, :]))

    # wiener deblur
    x_deblur = torch.zeros(x.size()).cuda()
    ks = kernel_norm.shape[2]
    dim = (ks, ks, ks, ks)
    x_pad = F.pad(x, dim, "replicate")  # padding

    for i in range(x_pad.shape[1]):
        # blur feature channel
        x_pad_ch = x_pad[:, i:i + 1, :, :]
        x_deblur_ch = wiener_deblur_func(
            x_pad_ch, kernel_norm)  # feature domain winner deconvolution
        x_deblur[:, i:i + 1, :, :] = x_deblur_ch[:,
                                                 :, ks:-ks, ks:-ks]  # remove padding
    return x_deblur


class FeatModule(nn.Module):
    def __init__(self, n_in, n_feats=16, kernel_size=5, padding=2, act=True):
        super(FeatModule, self).__init__()
        FeatureBlock = [Conv(n_in, n_feats, kernel_size, padding=padding, act=act),
                        ResBlock(Conv, n_feats, kernel_size, padding=padding),
                        ResBlock(Conv, n_feats, kernel_size, padding=padding),
                        ResBlock(Conv, n_feats, kernel_size, padding=padding)]
        self.FeatureBlock = nn.Sequential(*FeatureBlock)

    def forward(self, x):
        return self.FeatureBlock(x)

if __name__ == '__main__':
    pass

