import sys
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader, DistributedSampler, random_split
from scipy import ndimage
import cv2
import os
import numpy as np
from tqdm import tqdm
from os.path import join as opj
from srcs.utils import utils_deblur_kair
from srcs.utils.utils_image_zzh import augment_img
from srcs.utils.utils_noise_zzh import BasicNoiseModel, CMOS_Camera

# =================
# loading single frame and blur it
# =================

# =================
# basic functions
# =================


def init_network_input(coded_blur_img, code):
    """
    calculate the initial input of the network

    Args:
        coded_blur_img (ndarray): coded measurement
        code (ndarray): encoding code
    """
    return coded_blur_img


def img_saturation(img, mag_times=1.2, min=0, max=1):
    """
    saturation generation by magnify and clip
    """
    # return np.clip(img*mag_times, min, max)
    return np.clip(img*mag_times, min, max)/mag_times

# =================
# Dataset
# =================


class BlurImgDataset(Dataset):
    """
    generate blurry images from loaded sharp images, load samples during each iteration
    """

    def __init__(self, img_dir, ce_code=None, patch_sz=256, tform_op=None, noise_type='gaussian', noise_params={'sigma': 0}, motion_len=0, load_psf_dir=None, test_mode='one2part'):
        super(BlurImgDataset, self).__init__()
        self.ce_code = ce_code
        self.patch_sz = [patch_sz] * \
            2 if isinstance(patch_sz, int) else patch_sz
        self.tform_op = tform_op
        self.noise_type = noise_type
        self.motion_len = motion_len
        # use loaded psf, rather than generated
        self.load_psf = True if load_psf_dir else False
        self.img_paths = []
        self.img_num = None
        self.test_mode = test_mode
        self.noise_params = noise_params
        if noise_type == 'gaussian':
            self.noise_model = BasicNoiseModel(noise_type, noise_params)
        elif noise_type == 'camera':
            self.noise_model = CMOS_Camera(noise_params)
        self.motion_len = motion_len

        # get image paths and load images
        img_paths = []
        if isinstance(img_dir, str):
            # single dataset
            img_names = sorted(os.listdir(img_dir))
            img_paths = [opj(img_dir, img_name) for img_name in img_names]
        else:
            # multiple dataset
            for img_dir_n in sorted(img_dir):
                img_names_n = sorted(os.listdir(img_dir_n))
                img_paths_n = [opj(img_dir_n, img_name_n)
                               for img_name_n in img_names_n]
                img_paths.extend(img_paths_n)
        self.img_paths = img_paths

        for img_path in self.img_paths:
            if img_path.split('.')[-1].lower() not in ['jpg', 'jpeg', 'png', 'tif', 'bmp']:
                print('Skip a non-image file: %s' % (img_path))
                self.img_paths.remove(img_path)
        self.img_num = len(self.img_paths)
        print('===> dataset image num: %d' % self.img_num)

        # get loaded psf paths and load psfs
        if self.load_psf:
            psf_names = sorted(os.listdir(load_psf_dir))
            self.psf_paths = [opj(load_psf_dir, psf_name)
                              for psf_name in psf_names]
            for psf_path in self.psf_paths:
                if psf_path.split('.')[-1].lower() not in ['jpg', 'jpeg', 'png', 'tif', 'bmp']:
                    print('Skip a non-image file:%s' % (psf_path))
                    self.psf_paths.remove(psf_path)
            self.psf_num = len(psf_names)
            print('===> dataset psf num: %d' % self.psf_num)

    def __getitem__(self, idx):
        # index for load image and psf
        img_idx = idx
        if self.load_psf:
            if self.test_mode == 'one2part':
                if self.psf_num < self.img_num:
                    psf_idx = idx*self.psf_num//self.img_num
                else:
                    psf_idx = idx
                img_idx = idx
            elif self.test_mode == 'one2all':
                psf_idx = idx//self.img_num
                img_idx = idx % self.img_num
            else:
                raise NotImplementedError

        # load image
        imgk = cv2.imread(self.img_paths[img_idx])
        assert imgk is not None, 'Image-%s read falied' % self.img_paths[img_idx]
        imgk = cv2.cvtColor(imgk, cv2.COLOR_BGR2RGB).astype(np.float32)/255
        img_sz = imgk.shape

        # crop to patch size
        if self.patch_sz:
            assert (img_sz[0] >= self.patch_sz[0]) and (img_sz[1] >= self.patch_sz[1]
                                                        ), 'error patch_size(%d*%d) larger than image size(%d*%d)' % (*self.patch_sz, *img_sz[0:2])
            xmin = np.random.randint(0, img_sz[1]-self.patch_sz[1])
            ymin = np.random.randint(0, img_sz[0]-self.patch_sz[0])
            imgk = imgk[ymin:ymin+self.patch_sz[0],
                        xmin:xmin+self.patch_sz[1], :]
        # data augment
        if self.tform_op:
            imgk = augment_img(imgk, tform_op=self.tform_op)

        # get/load psf
        if self.load_psf:
            psfk = cv2.imread(self.psf_paths[psf_idx])
            assert psfk is not None, 'PSF-%s read falied' % self.psf_paths[img_idx]
            psfk = cv2.cvtColor(psfk, cv2.COLOR_BGR2GRAY)
            psfk = psfk.astype(np.float32)/np.sum(psfk)
        else:
            psfk = utils_deblur_kair.blurkernel_synthesis_zzh(h=37)

        # energy normalize
        if self.ce_code:
            psfk = psfk*sum(self.ce_code)/len(self.ce_code)

        # convolve image with psf
        coded_blur_img = ndimage.filters.convolve(
            imgk, np.expand_dims(psfk, axis=2), mode='wrap').astype(np.float32)

        # add noise
        if self.noise_type == 'gaussian':
            coded_blur_img_noisy, _ = self.noise_model.add_noise(
                coded_blur_img)
        elif self.noise_type == 'camera':
            self.noise_model.uniform_sampling_noise_params(
                self.noise_params)  # uniform sampling parameters
            kc = self.noise_params['kc']
            kc = kc if isinstance(kc, (int, float)) else np.floor(np.random.uniform(
                *kc))
            coded_blur_img_noisy = self.noise_model.take_photo_P(
                coded_blur_img*255, imgsize=coded_blur_img.shape, kd=kc/6, ka=6)/255

        # saturation
        coded_blur_img_noisy = img_saturation(coded_blur_img_noisy)

        # data arrange
        psfk = np.expand_dims(np.float32(psfk), axis=2)
        # noise_level = np.reshape(noise_level, [1, 1, 1]).astype(np.float32)
        coded_blur_img_noisy = coded_blur_img_noisy.astype(np.float32)

        # return [C,H,W]
        # noise_level = -1, i.e. don't return it
        return coded_blur_img_noisy.transpose(2, 0, 1), psfk.transpose(2, 0, 1), imgk.transpose(2, 0, 1), coded_blur_img.transpose(2, 0, 1)

    def __len__(self):
        return self.img_num


class BlurImgDataset_all2CPU(Dataset):
    """
    generate blurry images from loaded sharp images, load entire dataset to CPU to speed the data load process
    test mode: 
        - one2all: every kernel maps to every image
        - one2part: every kernel maps to a part of images (img_num/kernel_num)
    """

    def __init__(self, img_dir, ce_code=None, patch_sz=256, tform_op=None, noise_type='gaussian', noise_params={'sigma': 0}, motion_len=0, load_psf_dir=None, test_mode='one2part'):
        super(BlurImgDataset_all2CPU, self).__init__()
        self.ce_code = ce_code
        self.patch_sz = [patch_sz] * \
            2 if isinstance(patch_sz, int) else patch_sz
        self.tform_op = tform_op
        self.noise_type = noise_type
        self.noise_params = noise_params
        if noise_type == 'gaussian':
            self.noise_model = BasicNoiseModel(noise_type, noise_params)
        elif noise_type == 'camera':
            self.noise_model = CMOS_Camera(noise_params)
        self.motion_len = motion_len
        # use loaded psf, rather than generated
        self.load_psf = True if load_psf_dir else False
        self.test_mode = test_mode
        self.img_paths = []
        self.imgs = []
        self.psfs = []
        self.img_num = None

        # get image paths and load images
        img_paths = []
        if isinstance(img_dir, str):
            # single dataset
            img_names = sorted(os.listdir(img_dir))
            img_paths = [opj(img_dir, img_name) for img_name in img_names]
        else:
            # multiple dataset
            for img_dir_n in sorted(img_dir):
                img_names_n = sorted(os.listdir(img_dir_n))
                img_paths_n = [opj(img_dir_n, img_name_n)
                               for img_name_n in img_names_n]
                img_paths.extend(img_paths_n)
        self.img_paths = img_paths

        for img_path in tqdm(self.img_paths, desc='Loading image to CPU'):

            if img_path.split('.')[-1].lower() not in ['jpg', 'jpeg', 'png', 'tif', 'bmp']:
                print('Skip a non-image file: %s' % (img_path))
                continue
            img = cv2.imread(img_path)
            assert img is not None, 'Image read falied'
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            self.imgs.append(img)

        self.img_num = len(self.imgs)
        self.blur_img_num = self.img_num

        # get loaded psf paths and load psfs
        if self.load_psf:
            psf_names = sorted(os.listdir(load_psf_dir))
            self.psf_num = len(psf_names)

            for psf_name in tqdm(psf_names, desc='Loading psf to CPU'):
                psf_path = opj(load_psf_dir, psf_name)
                if psf_path.split('.')[-1].lower() not in ['jpg', 'jpeg', 'png', 'tif', 'bmp']:
                    print('Skip a non-image file: %s' % psf_path)
                    continue
                psf = cv2.imread(psf_path)
                assert psf is not None, 'Image read falied'
                psf = cv2.cvtColor(psf, cv2.COLOR_BGR2GRAY)
                psf = psf.astype(np.float32)/np.sum(psf)  # normalized to sum=1
                self.psfs.append(psf)

            if self.test_mode == 'one2all':
                self.blur_img_num = self.psf_num*self.img_num
            elif self.test_mode == 'one2part':
                self.blur_img_num = self.img_num
            else:
                raise NotImplementedError

    def __getitem__(self, idx):
        # index for load psf
        img_idx = idx
        if self.load_psf:
            if self.test_mode == 'one2part':
                if self.psf_num < self.img_num:
                    psf_idx = idx*self.psf_num//self.img_num
                else:
                    psf_idx = idx
                img_idx = idx
            elif self.test_mode == 'one2all':
                psf_idx = idx//self.img_num
                img_idx = idx % self.img_num
            else:
                raise NotImplementedError

        # load sharp image
        imgk = np.array(self.imgs[img_idx], dtype=np.float32)/255
        img_sz = imgk.shape
        # crop to patch size
        if self.patch_sz:
            if (img_sz[0] < self.patch_sz[0]) or (img_sz[1] < self.patch_sz[1]):
                print('PATCH_SZ(%d*%d) larger than image size(%d*%d), use a grey image' %
                      (*self.patch_sz, *img_sz[0:2]))
                imgk = np.ones(
                    (*self.patch_sz, img_sz[2])).astype(np.float32)*0.5
                img_sz = imgk.shape
            x_margin = img_sz[1]-self.patch_sz[1]
            y_margin = img_sz[0]-self.patch_sz[0]
            xmin = np.random.randint(0, x_margin) if x_margin != 0 else 0
            ymin = np.random.randint(0, y_margin) if y_margin != 0 else 0
            imgk = imgk[ymin:ymin+self.patch_sz[0],
                        xmin:xmin+self.patch_sz[1], :]

        # data augment
        if self.tform_op:
            imgk = augment_img(imgk, tform_op=self.tform_op)

        # get psf, noise level and calc blur image
        if self.load_psf:
            # load psf
            psfk = self.psfs[psf_idx]
            # psf = psf[1:,1:] # for odd size psf
        else:
            psfk = utils_deblur_kair.blurkernel_synthesis_zzh(h=37)

        # energy normalize
        if self.ce_code:
            psfk = psfk*sum(self.ce_code)/len(self.ce_code)

        # convolve image with psf
        coded_blur_img = ndimage.filters.convolve(
            imgk, np.expand_dims(psfk, axis=2), mode='wrap').astype(np.float32)

        # add noise
        if self.noise_type == 'gaussian':
            coded_blur_img_noisy, _ = self.noise_model.add_noise(
                coded_blur_img)
        elif self.noise_type == 'camera':
            self.noise_model.uniform_sampling_noise_params(
                self.noise_params)  # uniform sampling parameters
            kc = self.noise_params['kc']
            kc = kc if isinstance(kc, (int, float)) else np.floor(np.random.uniform(
                *kc))
            coded_blur_img_noisy = self.noise_model.take_photo_P(
                coded_blur_img*255, imgsize=coded_blur_img.shape, kd=kc/6, ka=6)/255

        # saturation
        coded_blur_img_noisy = img_saturation(coded_blur_img_noisy)

        # data arrange
        psfk = np.expand_dims(np.float32(psfk), axis=2)
        # noise_level = np.reshape(noise_level, [1, 1, 1]).astype(np.float32)
        coded_blur_img_noisy = coded_blur_img_noisy.astype(np.float32)

        # return [C,H,W]
        # noise_level = -1, i.e. don't return it
        return coded_blur_img_noisy.transpose(2, 0, 1), psfk.transpose(2, 0, 1), imgk.transpose(2, 0, 1), coded_blur_img.transpose(2, 0, 1)

    def __len__(self):
        return self.blur_img_num


class BlurImgDataset_Exp_all2CPU(Dataset):
    """
    load blurry image, kernel, and ground truth (for 'simuexp' exp) for normal experiments, load entire dataset to CPU to speed the data load process. (image format data)
    exp_mode:
        - simuexp: with gt
        - realexp: no gt   
    patch_sz: assign image size of patch processing to save GPU memory, default = None, use whole image (TODO: patch processing and stitching)
    """

    def __init__(self, blur_img_dir, psf_dir, gt_dir=None, patch_sz=None, exp_mode='simuexp'):
        super(BlurImgDataset_Exp_all2CPU, self).__init__()
        self.patch_sz = [patch_sz] * \
            2 if isinstance(patch_sz, int) else patch_sz
        # use loaded psf, rather than generated
        self.img_dir, self.psf_dir, self.gt_dir = blur_img_dir, psf_dir, gt_dir
        self.exp_mode = exp_mode
        self.imgs = []
        self.psfs = []
        self.gts = []

        # get image paths and load images
        img_paths = []
        img_names = sorted(os.listdir(blur_img_dir))
        img_paths = [opj(blur_img_dir, img_name) for img_name in img_names]
        self.img_num = len(img_paths)

        for img_path in tqdm(img_paths, desc='Loading image to CPU'):

            if img_path.split('.')[-1].lower() not in ['jpg', 'jpeg', 'png', 'tif', 'bmp']:
                print('Skip a non-image file: %s' % (img_path))
                continue
            img = cv2.imread(img_path)
            assert img is not None, 'Image read falied'
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            self.imgs.append(img)

        # get gt paths and load gts
        if self.exp_mode == 'simuexp':
            gt_paths = []
            gt_names = sorted(os.listdir(gt_dir))
            gt_paths = [opj(gt_dir, gt_name) for gt_name in gt_names]
            self.gt_num = len(gt_paths)

            for gt_path in tqdm(gt_paths, desc='Loading gt to CPU'):
                if gt_path.split('.')[-1].lower() not in ['jpg', 'jpeg', 'png', 'tif', 'bmp']:
                    print('Skip a non-image file: %s' % (gt_path))
                    continue
                gt = cv2.imread(gt_path)
                assert gt is not None, 'Image read falied'
                gt = cv2.cvtColor(gt, cv2.COLOR_BGR2RGB)
                self.gts.append(gt)

        # get loaded psf paths and load psfs
        psf_paths = []
        psf_names = sorted(os.listdir(psf_dir))
        psf_paths = [opj(psf_dir, psf_name) for psf_name in psf_names]
        self.psf_num = len(psf_names)

        for psf_path in tqdm(psf_paths, desc='Loading psf to CPU'):
            if psf_path.split('.')[-1].lower() not in ['jpg', 'jpeg', 'png', 'tif', 'bmp']:
                print('Skip a non-image file: %s' % psf_path)
                continue
            psf = cv2.imread(psf_path)
            assert psf is not None, 'Image read falied'
            psf = cv2.cvtColor(psf, cv2.COLOR_BGR2GRAY)
            psf = psf.astype(np.float32)/np.sum(psf)  # normalized to sum=1
            self.psfs.append(psf)

    def real_data_preproc(self, img, kernel):
        # Reducing boundary artifacts in image deconvolution
        # img_warp = edgetaper_np(img, kernel)

        H, W = img.shape[0:2]
        H1, W1 = np.int32(H/2), np.int32(W/2)
        img_warp = np.pad(
            img/1.2, ((H1, H1), (W1, W1), (0, 0)), mode='symmetric')

        return img_warp

    def __getitem__(self, idx):
        # load psf
        psfk = np.array(self.psfs[idx], dtype=np.float32)

        # load blurry data
        _imgk = np.array(self.imgs[idx], dtype=np.float32)/255
        if self.exp_mode == 'simuexp':
            imgk = _imgk
            gtk = np.array(self.gts[idx], dtype=np.float32)/255

        elif self.exp_mode == 'realexp':
            # imgk = _imgk
            # imgk = np.expand_dims(_imgk, 2).repeat(3, 2)
            imgk = self.real_data_preproc(_imgk, psfk)
            gtk = np.zeros_like(imgk, dtype=np.float32)

        return imgk.transpose(2, 0, 1).astype(np.float32), psfk[np.newaxis, :], gtk.transpose(2, 0, 1), []

    def __len__(self):
        return self.img_num


# =================
# get dataloader
# =================

def get_data_loaders(img_dir=None, ce_code=None, patch_size=256, batch_size=8, tform_op=None, noise_type='gaussian', noise_params={'sigma': 0.05}, motion_len=[10, 20], load_psf_dir=None, load_blur_image_dir=None,  shuffle=True, validation_split=None, status='train', num_workers=8, pin_memory=False, prefetch_factor=2, all2CPU=True, test_mode='one2all'):
    # dataset
    if status in ['train', 'test', 'debug', 'valid']:
        if all2CPU:
            dataset = BlurImgDataset_all2CPU(
                img_dir, ce_code, patch_size, tform_op, noise_type, noise_params, motion_len, load_psf_dir=load_psf_dir, test_mode=test_mode)
        else:
            dataset = BlurImgDataset(
                img_dir, ce_code, patch_size, tform_op, noise_type, noise_params, motion_len, load_psf_dir=load_psf_dir, test_mode=test_mode)
    elif status in ['simuexp', 'realexp']:
        dataset = BlurImgDataset_Exp_all2CPU(
            load_blur_image_dir, load_psf_dir, img_dir, patch_sz=patch_size, exp_mode=status)

    loader_args = {
        'batch_size': batch_size,
        'shuffle': shuffle,
        'num_workers': num_workers,
        'prefetch_factor': prefetch_factor,
        'pin_memory': pin_memory
    }

    # dataset split & dist train assignment
    if status in ['train', 'debug', 'valid']:
        # split dataset into train and validation set
        num_total = len(dataset)
        if isinstance(validation_split, int):
            assert validation_split > 0
            assert validation_split < num_total, "validation set size is configured to be larger than entire dataset."
            num_valid = validation_split
        elif isinstance(validation_split, float):
            num_valid = int(num_total * validation_split)
        else:
            num_valid = 0  # don't split valid set

        num_train = num_total - num_valid

        train_dataset, valid_dataset = random_split(
            dataset, [num_train, num_valid])

        # distribution trainning setting
        train_sampler, valid_sampler = None, None
        if dist.is_initialized():
            loader_args['shuffle'] = False
            train_sampler = DistributedSampler(train_dataset)
            if num_valid != 0:
                valid_sampler = DistributedSampler(valid_dataset)

        train_dataloader = DataLoader(
            train_dataset, sampler=train_sampler, **loader_args)
        if num_valid != 0:
            val_dataloader = DataLoader(
                valid_dataset, sampler=valid_sampler, **loader_args)
        else:
            val_dataloader = []

        return train_dataloader, val_dataloader

    elif status in ['test', 'realexp', 'simuexp']:
        return DataLoader(dataset, **loader_args)

    elif status == 'valid':
        if dist.is_initialized():
            loader_args['shuffle'] = False
            sampler = DistributedSampler(dataset)
        return DataLoader(dataset, sampler=sampler, **loader_args)
    else:
        raise(ValueError(
            "$Status can only be 'train'|'debug'|'test'|'valid'|'simuexp'|'realexp'"))


if __name__ == '__main__':
    sys.path.append(os.path.dirname(__file__) + os.sep + '../')
    # from srcs.utils import utils_blurkernel_zzh
    from utils import utils_deblur_kair
    from utils.utils_noise_zzh import BasicNoiseModel, CMOS_Camera
    from utils.utils_image_zzh import augment_img

    img_dir = 'dataset/train_data/Kodak24/'

    # load_psf_dir = '/hdd/1/zzh/project/INFWIDE/dataset/benchmark/Sun_dataset/kernel/'
    psf_dir = None

    save_dir = './outputs/tmp/test/'

    # train_dataloader, val_dataloader = get_data_loaders(
    #     img_dir, ce_code=None, patch_size=256, tform_op=['all'], sigma_range=0, motion_len=[25, 48], batch_size=1, num_workers=8, all2CPU=False)

    # test_dataloader = get_data_loaders(
    #     img_dir, patch_size=None, load_psf_dir=load_psf_dir, noise_type='gaussian', noise_params={'sigma': [0,0.03]},  motion_len=[25, 48], batch_size=1, num_workers=8, shuffle=False, all2CPU=True, status='test')
    test_dataloader = get_data_loaders(
        img_dir, patch_size=None, load_psf_dir=psf_dir, noise_type='camera', noise_params={'sigma_beta': [0, 0.02], 'sigma_r': [0, 0.03], 'nd_factor': [1, 5], 'kc': [5, 20]},  motion_len=[25, 48], batch_size=1, num_workers=8, shuffle=False, all2CPU=False, status='test')

    iter_dataloader = test_dataloader

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    k = 0

    for coded_blur_img_noisy, psf, sharp_img, coded_blur_img, noise_level in iter_dataloader:  # val_dataloader
        k += 1
        coded_blur_img_noisy = coded_blur_img_noisy.numpy(
        )[0, ::-1, ...].transpose(1, 2, 0)*255
        coded_blur_img = coded_blur_img.numpy(
        )[0, ::-1, ...].transpose(1, 2, 0)*255
        sharp_img = sharp_img.numpy()[0, ::-1, ...].transpose(1, 2, 0)*255
        psf = psf.numpy()[0].transpose(1, 2, 0)
        psf = psf/np.max(psf)*255

        # import matplotlib.pyplot as plt
        # plt.imshow(psf, interpolation="nearest", cmap="gray")
        # plt.show()

        if k % 1 == 0:
            print('k = ', k)
            cv2.imwrite(opj(save_dir, 'coded_blur_img_noisy%02d.png' %
                        k), coded_blur_img_noisy, [cv2.IMWRITE_PNG_COMPRESSION, 0])
            cv2.imwrite(opj(save_dir, 'coded_blur_img%02d.png' %
                        k), coded_blur_img, [cv2.IMWRITE_PNG_COMPRESSION, 0])
            cv2.imwrite(opj(save_dir, 'clear%02d.png' %
                        k), sharp_img, [cv2.IMWRITE_PNG_COMPRESSION, 0])
            cv2.imwrite(opj(save_dir, 'psf%02d.png' %
                        k), psf, [cv2.IMWRITE_PNG_COMPRESSION, 0])
