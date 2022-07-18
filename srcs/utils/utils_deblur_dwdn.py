
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair, _quadruple

'''
This is based on the implementation by Kai Zhang (github: https://github.com/cszn)
'''

# --------------------------------
# --------------------------------


class MedianPool2d(nn.Module):
    """ Median pool (usable as median filter when stride=1) module.

    Args:
         kernel_size: size of pooling kernel, int or 2-tuple
         stride: pool stride, int or 2-tuple
         padding: pool padding, int or 4-tuple (l, r, t, b) as in pytorch F.pad
         same: override padding and enforce same padding, boolean
    """

    def __init__(self, kernel_size=3, stride=1, padding=0, same=False):
        super(MedianPool2d, self).__init__()
        self.k = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _quadruple(padding)  # convert to l, r, t, b
        self.same = same

    def _padding(self, x):
        if self.same:
            ih, iw = x.size()[2:]
            if ih % self.stride[0] == 0:
                ph = max(self.k[0] - self.stride[0], 0)
            else:
                ph = max(self.k[0] - (ih % self.stride[0]), 0)
            if iw % self.stride[1] == 0:
                pw = max(self.k[1] - self.stride[1], 0)
            else:
                pw = max(self.k[1] - (iw % self.stride[1]), 0)
            pl = pw // 2
            pr = pw - pl
            pt = ph // 2
            pb = ph - pt
            padding = (pl, pr, pt, pb)
        else:
            padding = self.padding
        return padding

    def forward(self, x):
        # using existing pytorch functions and tensor ops so that we get autograd,
        # would likely be more efficient to implement from scratch at C/Cuda level
        x = F.pad(x, self._padding(x), mode='reflect')
        x = x.unfold(2, self.k[0], self.stride[0]).unfold(
            3, self.k[1], self.stride[1])
        x = x.contiguous().view(x.size()[:4] + (-1,)).median(dim=-1)[0]
        return x

# --------------------------------
# --------------------------------


def get_uperleft_denominator(img, kernel):
    # discrete fourier transform of kernel
    ker_f = convert_psf2otf(kernel, img.size()[-2:])
    nsr = wiener_filter_para(img)
    denominator = inv_fft_kernel_est(ker_f, nsr)
    img1 = img.cuda()
    # numerator = torch.rfft(img1, 3, onesided=False)  # zzh: for torch<=1.7
    # zzh: for torch>1.7
    numerator = torch.fft.fft2(img1, dim=(-2, -1))
    numerator = torch.stack((numerator.real, numerator.imag), -1)
    deblur = deconv(denominator, numerator)
    return deblur

# --------------------------------
# --------------------------------


def wiener_filter_para(_input_blur):
    median_filter = MedianPool2d(
        kernel_size=3, padding=1)(_input_blur)
    diff = median_filter - _input_blur
    num = (diff.shape[2]*diff.shape[2])
    mean_n = torch.sum(diff, (2, 3)).view(-1, 1, 1, 1)/num
    var_n = torch.sum((diff - mean_n) * (diff - mean_n),
                      (2, 3))/(num-1)

    mean_input = torch.sum(_input_blur, (2, 3)).view(-1, 1, 1, 1)/num
    var_s2 = (torch.sum((_input_blur-mean_input) *
                        (_input_blur-mean_input), (2, 3))/(num-1))**(0.5)
    NSR = var_n / var_s2 * 8.0 / 3.0 / 10.0
    NSR = NSR.view(-1, 1, 1, 1)
    return NSR


# --------------------------------
# --------------------------------


def inv_fft_kernel_est(ker_f, NSR):
    inv_denominator = ker_f[:, :, :, :, 0] * ker_f[:, :, :, :, 0] \
        + ker_f[:, :, :, :, 1] * ker_f[:, :, :, :, 1] + NSR
    # pseudo inverse kernel in flourier domain.
    inv_ker_f = torch.zeros_like(ker_f)
    inv_ker_f[:, :, :, :, 0] = ker_f[:, :, :, :, 0] / inv_denominator
    inv_ker_f[:, :, :, :, 1] = -ker_f[:, :, :, :, 1] / inv_denominator
    return inv_ker_f

# --------------------------------
# --------------------------------


def deconv(inv_ker_f, fft_input_blur):
    # delement-wise multiplication.
    deblur_f = torch.zeros_like(inv_ker_f).cuda()
    deblur_f[:, :, :, :, 0] = inv_ker_f[:, :, :, :, 0] * fft_input_blur[:, :, :, :, 0] \
        - inv_ker_f[:, :, :, :, 1] * fft_input_blur[:, :, :, :, 1]
    deblur_f[:, :, :, :, 1] = inv_ker_f[:, :, :, :, 0] * fft_input_blur[:, :, :, :, 1] \
        + inv_ker_f[:, :, :, :, 1] * fft_input_blur[:, :, :, :, 0]
    # deblur = torch.irfft(deblur_f, 3, onesided=False)  # zzh: for torch<=1.7
    # zzh: for torch>1.7
    deblur_f = deblur_f[:, :, :, :, 0]+deblur_f[:, :, :, :, 1]*1j
    deblur = torch.fft.ifft2(deblur_f, dim=(-2, -1))
    return deblur

# --------------------------------
# --------------------------------


# modified from utils__deblur_kair: p2o
def convert_psf2otf(psf, shape, device='cuda:0'):
    '''
    # psf: NxCxhxw
    # shape: [H,W]
    # otf: NxCxHxWx2
    '''
    otf = torch.zeros(psf.shape[:-2] + shape).type_as(psf)
    otf[..., :psf.shape[2], :psf.shape[3]].copy_(psf)
    for axis, axis_size in enumerate(psf.shape[2:]):
        otf = torch.roll(otf, -int(axis_size / 2), dims=axis+2)
    # zzh: for torch<=1.7
    # otf = torch.rfft(otf, 3, onesided=False)
    # zzh: for torch>1.7
    otf = torch.fft.fft2(otf, dim=(-2, -1))
    otf = torch.stack((otf.real, otf.imag), -1)
    # n_ops = torch.sum(torch.tensor(psf.shape).type_as(
    #     psf) * torch.log2(torch.tensor(psf.shape).type_as(psf)))
    # otf[..., 1][torch.abs(otf[..., 1]) < n_ops *
    #             2.22e-16] = torch.tensor(0).type_as(psf)
    return otf.to(device)


def postprocess(*images, rgb_range):
    def _postprocess(img):
        pixel_range = 255 / rgb_range
        return img.mul(pixel_range).clamp(0, 255).round().div(pixel_range)

    return [_postprocess(img) for img in images]
