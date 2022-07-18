import sys
import cv2
import torch
import numpy as np
from os.path import join as opj
import itertools

# ===============
# image/video transform
# ===============


def augment_img(img, prob=0.5, tform_op=['all']):
    """
    img data augment with a $op chance

    Args:
        img ([ndarray]): [shape: H*W*C]
        prob (float, optional): [probility]. Defaults to 0.5.
        op (list, optional): ['flip' | 'rotate' | 'reverse']. Defaults to ['all'].
    """
    if 'flip' in tform_op or 'all' in tform_op:
        # flip left-right or flip up-down
        if np.random.rand() < prob:
            img = img[:, ::-1, :]
        if np.random.rand() < prob:
            img = img[::-1, :, :]
    if 'rotate' in tform_op or 'all' in tform_op:
        # rotate 90 / -90 degrees
        if prob/4 < np.random.rand() <= prob/2:
            np.transpose(img, axes=(1, 0, 2))[::-1, ...]  # -90
        elif prob/2 < np.random.rand() <= prob:
            img = np.transpose(
                img[::-1, :, :][:, ::-1, :], axes=(1, 0, 2))[::-1, ...]  # 90

    return img.copy()


def augment_vid(vid, prob=0.5, tform_op=['all']):
    """
    video data transform (data augment) with a $op chance

    Args:
        vid ([ndarray]): [shape: N*H*W*C]
        prob (float, optional): [probility]. Defaults to 0.5.
        op (list, optional): ['flip' | 'rotate' | 'reverse']. Defaults to ['all'].
    """
    if 'flip' in tform_op or 'all' in tform_op:
        # flip left-right or flip up-down
        if np.random.rand() < prob:
            vid = vid[:, :, ::-1, :]
        if np.random.rand() < prob:
            vid = vid[:, ::-1, :, :]
    if 'rotate' in tform_op or 'all' in tform_op:
        # rotate 90 / -90 degrees
        if prob/4 < np.random.rand() <= prob/2:
            np.transpose(vid, axes=(0, 2, 1, 3))[:, ::-1, ...]  # -90
        elif prob/2 < np.random.rand() <= prob:
            vid = np.transpose(
                vid[:, ::-1, :, :][:, :, ::-1, :], axes=(0, 2, 1, 3))[:, ::-1, ...]  # 90

# ===============
# padding
# ===============


def pad_circular(x, pad):
    """

    :param x: shape [H, W]
    :param pad: int >= 0
    :return:
    """
    x = torch.cat([x, x[0:pad]], dim=0)
    x = torch.cat([x, x[:, 0:pad]], dim=1)
    x = torch.cat([x[-2 * pad:-pad], x], dim=0)
    x = torch.cat([x[:, -2 * pad:-pad], x], dim=1)

    return x


def pad_circular_nd(x: torch.Tensor, pad: int, dim) -> torch.Tensor:
    """
    :param x: shape [H, W]
    :param pad: int >= 0
    :param dim: the dimension over which the tensors are padded
    :return:
    """

    if isinstance(dim, int):
        dim = [dim]

    for d in dim:
        if d >= len(x.shape):
            raise IndexError(f"dim {d} out of range")

        idx = tuple(slice(0, None if s != d else pad, 1)
                    for s in range(len(x.shape)))
        x = torch.cat([x, x[idx]], dim=d)

        idx = tuple(slice(None if s != d else -2 * pad, None if s !=
                    d else -pad, 1) for s in range(len(x.shape)))
        x = torch.cat([x[idx], x], dim=d)
        pass

    return x



# ===============
# show and save
# ===============


def img_matrix(imgs, n_row, n_col, margin=0):
    '''
        Arrange a number of images as a matrix.

        positional arguments:
        imgs        N images [H,W,C] in a list
        n_row       number of rows in desired images matrix
        n_col       number of columns in desired images matrix
        margin      Margin between images: integers are interpreted as pixels,floats as proportions. default=0

        reference: https://gist.github.com/pgorczak/95230f53d3f140e4939c
    '''

    n = n_col*n_row
    if len(imgs) != n:
        raise ValueError('Number of images ({}) does not match '
                         'matrix size {}x{}'.format(n_col, n_row, len(imgs)))

    if any(i.shape != imgs[0].shape for i in imgs[1:]):
        raise ValueError('Not all images have the same shape.')

    img_h, img_w = imgs[0].shape[0:2]
    if imgs[0].ndim == 2:
        img_c = 1
    else:
        img_c = imgs[0].shape[2]

    m_x = 0
    m_y = 0

    if isinstance(margin, float):
        m = float(margin)
        m_x = int(m*img_w)
        m_y = int(m*img_h)
    else:
        m_x = int(margin)
        m_y = m_x

    imgmatrix = np.zeros((img_h * n_row + m_y * (n_row - 1),
                          img_w * n_col + m_x * (n_col - 1),
                          img_c),
                         np.uint8)
    imgmatrix = imgmatrix.squeeze()

    imgmatrix.fill(255)

    positions = itertools.product(range(n_col), range(n_row))
    for (x_i, y_i), img in zip(positions, imgs):
        x = x_i * (img_w + m_x)
        y = y_i * (img_h + m_y)
        imgmatrix[y:y+img_h, x:x+img_w,...] = img

    return imgmatrix
