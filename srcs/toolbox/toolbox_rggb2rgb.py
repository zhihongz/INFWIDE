import sys
import cv2
import torch
import numpy as np
from os.path import join as opj
import os

# ===============
# image format convert
# ===============


def rggb_demosaic(rggb, H, W, mode='BGGR'):
    out = np.zeros([H, W], dtype=np.float32)
    out[0::2, 0::2] = rggb[0]
    out[1::2, 0::2] = rggb[1]
    out[0::2, 1::2] = rggb[2]
    out[1::2, 1::2] = rggb[3]

    min_x, max_x = np.min(np.ravel(out)), np.max(np.ravel(out))
    x1 = np.uint8((out - min_x)/(max_x - min_x) * (2**8 - 1))

    if mode == 'BGGR':
        x_rgb = cv2.cvtColor(x1, cv2.COLOR_BayerRG2BGR)
        x_rgb = x_rgb.astype('float32')*(max_x - min_x)/(2**8 - 1) + min_x
        x_rgb[:, :, 0] *= 2310/1024
        x_rgb[:, :, 2] *= 1843/1024

    else:
        raise(NotImplemented)

    return x_rgb


# =============
# main function
# =============

if __name__ == '__main__':
    sys.path.append(os.path.dirname(__file__) + os.sep + '../')
    # path
    src_path1 = '/hdd/1/zzh/project/LLDeblur/outputs/benchmark/hu/realexp/y_rg1b_20_deblur.png'
    src_path2 = '/hdd/1/zzh/project/LLDeblur/outputs/benchmark/hu/realexp/y_rg2b_20_deblur.png'
    dst_path = '/hdd/1/zzh/project/LLDeblur/outputs/benchmark/hu/realexp/y_rgb_deblur.png'

# load
    bg1r = np.float32(cv2.imread(src_path1))
    bg2r = np.float32(cv2.imread(src_path2))

    h, w, c = bg1r.shape
    bggr = np.zeros((4, h, w))

    bggr[0] = (bg1r[..., 0]+bg2r[..., 0])/2
    bggr[1] = bg1r[..., 1]
    bggr[2] = bg2r[..., 1]
    bggr[3] = (bg1r[..., 2]+bg2r[..., 2])/2


    demosaic = rggb_demosaic(bggr, 2*h, 2*w, mode='BGGR')

    
    cv2.imwrite(dst_path, demosaic, [cv2.IMWRITE_PNG_COMPRESSION, 0])

