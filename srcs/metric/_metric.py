import torch
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim

## Note: the functions below takes 'torch tensor' as inputs


def accuracy(output, target):
    '''
    calculate classification accuracy
    '''
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        assert len(pred) == len(target)
        correct = torch.sum(pred == target).item()
    return correct / len(target)


def top_k_acc(output, target, k=3):
    '''
    calculate top-K classification accuracy
    '''
    with torch.no_grad():
        pred = torch.topk(output, k, dim=1)[1]
        assert len(pred) == len(target)
        correct = 0
        for i in range(k):
            correct += torch.sum(pred[:, i] == target).item()
    return correct / len(target)


def calc_psnr(output, target):
    '''
    calculate psnr
    '''
    output = output.cpu().detach().numpy()
    target = target.cpu().detach().numpy()
    assert output.shape[0] == target.shape[0]
    total_psnr = np.zeros(output.shape[0])
    for k, (pred, gt) in enumerate(zip(output, target)):
        if pred.ndim == 3:
            pred = pred.transpose(1, 2, 0)
            gt = gt.transpose(1, 2, 0)
        total_psnr[k] = compare_psnr(pred, gt, data_range=1)
    return np.mean(total_psnr)


def calc_ssim(output, target):
    '''
    calculate ssim
    '''
    output = output.cpu().detach().numpy()
    target = target.cpu().detach().numpy()
    assert output.shape[0] == target.shape[0]
    total_ssim = np.zeros(output.shape[0])
    for k, (pred, gt) in enumerate(zip(output, target)):
        if pred.ndim == 3:
            pred = pred.transpose(1, 2, 0)
            gt = gt.transpose(1, 2, 0)
        total_ssim[k] = compare_ssim(
            pred, gt, data_range=1, multichannel=True)
    return np.mean(total_ssim)
