from torch.autograd import Variable
import torch.nn as nn
from torchvision.models import vgg16_bn
import torch
import torch.nn.functional as F
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM

# ===========================
# weighted_loss
# ===========================


def weighted_loss(output, target, loss_weights_dict):
    loss = 0
    for k, v in loss_weights_dict.items():
        if k == 'l1_loss':
            loss += l1_loss(output, target)*v
        elif k == 'mse_loss':
            loss += mse_loss(output, target)*v
        elif k == 'perceptual_loss':
            loss += perceptual_loss(output, target)*v
        elif k == 'ssim_loss':
            loss += ssim_loss(output, target)*v
        elif k == 'msssim_loss':
            loss += msssim_loss(output, target)*v
        elif k == 'tv_loss':
            loss += tv_loss(output)*v
        else:
            raise NotImplementedError('No "%s" loss'%k)
    return loss


# ===========================
# basic_loss
# ===========================
def nll_loss(output, target):
    return F.nll_loss(output, target)


def mse_loss(output, target):
    return F.mse_loss(output, target)


def l1_loss(output, target):
    return F.l1_loss(output, target)


def ssim_loss(output, target):
    # Note: data range = 1
    ssim_loss = 1 - ssim(output, target,
                         data_range=1, size_average=True)
    return ssim_loss


def msssim_loss(output, target):
    # Note: data range = 1
    ms_ssim_loss = 1 - ms_ssim(output, target,
                               data_range=1, size_average=True)
    return ms_ssim_loss

def perceptual_loss(output, target):
    PL = PerceptualLoss()
    return PL(output, target)


def tv_loss(output, *args):
    # one input param, gt is not needed
    def _tensor_size(t):
        return t.size()[1]*t.size()[2]*t.size()[3]
    batch_size = output.size()[0]
    h_x = output.size()[2]
    w_x = output.size()[3]
    count_h = _tensor_size(output[:, :, 1:, :])
    count_w = _tensor_size(output[:, :, :, 1:])
    h_tv = torch.pow((output[:, :, 1:, :]-output[:, :, :h_x-1, :]), 2).sum()
    w_tv = torch.pow((output[:, :, :, 1:]-output[:, :, :, :w_x-1]), 2).sum()
    return 2*(h_tv/count_h+w_tv/count_w)/batch_size
