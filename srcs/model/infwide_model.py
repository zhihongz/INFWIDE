import torch.nn.functional as F
import torch.nn as nn
import torch
from srcs.model.infwide_modules_a import wiener_deblur_mc, FeatModule, image_nsr
from srcs.model.infwide_modules_b import FSRModule, XRFModule
from srcs.model._kair_denoiser_modules import ResUNet


# ===========================
# feature space wienner deconvolution network
# ===========================


class infwide(nn.Module):
    '''
    image and feature space multi-scale deconvolution network, cross residual fusion: 
    '''

    def __init__(self, n_colors, input_denoise=None):
        super(infwide, self).__init__()
        # params
        self.input_denoise = input_denoise

        self.FeatModule1 = FeatModule(
            n_in=n_colors, n_feats=16, kernel_size=5, padding=2, act=True)

        self.DenoiseUnet = ResUNet(
            in_nc=n_colors*2, out_nc=n_colors, nc=[32, 64, 128, 256])

        # Feature Space Refine
        self.FSRefineModule = FSRModule(
            n_colors=n_colors, n_resblock=3, n_in=16, n_feats=32, kernel_size=5, padding=2)  # DWDN type

        # Merge Refine
        self.MergeRefineModule = XRFModule(
            n_colors=n_colors, n_resblock=3, n_conv=3, n_feats=32, kernel_size=5, padding=2)

    def forward(self, img, kernel):
        ## Feature branch
        # get image feature
        feat = self.FeatModule1(img)
        # wiener deconv
        feat_wd = wiener_deblur_mc(feat, kernel)
        # refine & output
        feat_out = self.FSRefineModule(feat_wd)

        ## Image branch
        # image denoise
        if self.input_denoise == 'ResUnet':
            nsr = image_nsr(img).view(img.shape[0], img.shape[1], 1, 1).repeat(
                1, 1, img.shape[2], img.shape[3])
            img_denoise = self.DenoiseUnet(
                torch.cat((img, nsr), 1))  # denoise input
        else:
            img_denoise = img
        # wiener deconv
        img_wd = wiener_deblur_mc(img_denoise, kernel)  # wiener deblur

        ## fusion and refine
        out = self.MergeRefineModule(img_wd, feat_out)

        ## return
        return out, img_denoise
