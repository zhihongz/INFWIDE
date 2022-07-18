import numpy as np
import cv2

# =================
# Noise Model
# Various Noise Models
# ToDo:
# - support for grayscale image
#
# Update: 20220329
# Author: Zhihong Zhang, https://github.com/dawnlh
# =================

# ------------------
# basic functions
# ------------------


def uniform_pick(param):
    if isinstance(param, (int, float)):
        # a
        param_picked = param
    elif len(param) == 2:
        # [a1,a2]
        param_picked = np.random.uniform(*param)
    elif len(param) == 3:
        if isinstance(param[0], (int, float)):
            # [a,b,c]
            param_picked = param
        elif len(param[0]) == 2:
            # [[a1,a2],[b1,b2],[c1,c2]]
            param_picked = [0, 0, 0]
            param_picked[0] = np.random.uniform(
                *param[0])
            param_picked[1] = np.random.uniform(
                *param[1])
            param_picked[2] = np.random.uniform(
                *param[2])
        else:
            raise ValueError('parameter size wrong')
    else:
        raise ValueError('parameter size wrong')
    return param_picked


def gaussian_noise(clear_img, sigma):
    # awgn - gaussian noise model
    # clear_img (np.ndarray): clear image
    # sigma (scalar|list-range): guassian noise level
    if isinstance(sigma, (int, float)):
        noise_level = sigma
    else:
        noise_level = np.random.uniform(*sigma)
    img_noisy = clear_img + \
        np.random.normal(0, noise_level, clear_img.shape)
    return img_noisy, {'sigma': noise_level}


# --- modified from: Yuxiao Cheng: https://github.com/jarrycyx ---


class CMOS_Camera(object):
    def __init__(self, params):
        # params (dict): 'sigma_beta', 'sigma_r', 'nd_factor'
        self.uniform_sampling_noise_params(params)
        # self.sigma_beta = params['sigma_beta']
        # self.sigma_r = params['sigma_r']
        # self.nd_factor = params['nd_factor']

    def uniform_sampling_noise_params(self, noise_params):
        # generate a specific set of noise params from a range
        # noise_params (dict): noise parameters range
        def uniform_pick(param):
            if isinstance(param, (int, float)):
                # a
                param_picked = param
            elif len(param) == 2:
                # [a1,a2]
                param_picked = np.random.uniform(*param)
            elif len(param) == 3:
                if isinstance(param[0], (int, float)):
                    # [a,b,c]
                    param_picked = param
                elif len(param[0]) == 2:
                    # [[a1,a2],[b1,b2],[c1,c2]]
                    param_picked = [0, 0, 0]
                    param_picked[0] = np.random.uniform(
                        *param[0])
                    param_picked[1] = np.random.uniform(
                        *param[1])
                    param_picked[2] = np.random.uniform(
                        *param[2])
                else:
                    raise ValueError('parameter size wrong')
            else:
                raise ValueError('parameter size wrong')
            return param_picked

        self.sigma_beta = uniform_pick(noise_params['sigma_beta'])
        self.sigma_r = uniform_pick(noise_params['sigma_r'])
        self.nd_factor = uniform_pick(noise_params['nd_factor'])

    # sigma_beta, kc should be a list of size 3 or a number
    # imgsize should be a list of size 2 or 3
    # other arg should be a number
    # channel number of imgsize should match imgsource

    def simu_noise(self, img_source=None,
                   sigma_beta=0.01, nd_factor=1, sigma_r=1, ka=2,
                   exp_time=1, imgsize=[400, 800, 3], illumination=1,
                   sault_p=1e-5, pepper_p=1e-6, kd=3, gamma=1):

        def beta_c_array(sigma_beta, size=[200, 400]):
            beta_c_row = np.random.normal(
                loc=1, scale=sigma_beta, size=[size[0], 1])
            return np.array(beta_c_row).repeat(size[1], axis=1)

        def shot_noise(ne_array):
            return np.random.poisson(lam=ne_array)

        def dark_current(nd, size=[200, 400]):
            # return np.random.poisson(lam=nd, size=size)
            # return np.clip(np.random.exponential(nd, size=size) - nd, 0, np.inf)
            return np.clip(np.random.poisson(lam=nd, size=size) - nd, 0, np.inf)

        def read_noise(sigma_r, size=[200, 400]):
            return np.random.normal(loc=0, scale=sigma_r, size=size)

        def pepper_sault(img, p_pepper=0, p_sault=0):
            size = img.shape
            pepper = np.random.uniform(0, 1, size=size) < p_pepper
            sault = np.random.uniform(0, 1, size=size) < p_sault
            img[np.where(pepper == True)] = 0
            img[np.where(sault == True)] = 255
            return img

        green_gain = 2
        if img_source is None:
            ne_array = np.zeros(imgsize) * exp_time
        else:
            img_source = cv2.resize(
                img_source, (imgsize[1], imgsize[0])).astype(float)
            ne_array = img_source
            ne_array[:, :, 0] = img_source[:, :, 0] * exp_time * illumination
            ne_array[:, :, 1] = img_source[:, :, 1] * \
                exp_time * illumination * green_gain
            ne_array[:, :, 2] = img_source[:, :, 2] * exp_time * illumination

        if len(imgsize) == 3:
            ka = [ka, ka/green_gain, ka] if type(ka) != list else ka
            nd_factor = [nd_factor, nd_factor, nd_factor] if type(
                nd_factor) != list else nd_factor
            exp_time = [exp_time, exp_time, exp_time] if type(
                exp_time) != list else exp_time
            sigma_beta = [sigma_beta, sigma_beta, sigma_beta] if type(
                sigma_beta) != list else sigma_beta

            simu_img = np.zeros(imgsize)
            shot_noise_rgb = shot_noise(ne_array)
            for i in range(imgsize[2]):
                simu_img[:, :, i] = pepper_sault(ka[i] * beta_c_array(sigma_beta[i], size=imgsize) * (
                    shot_noise_rgb[:, :, i]
                    + dark_current(nd_factor[i] *
                                   exp_time[i], size=imgsize[:2])
                    + read_noise(sigma_r, size=imgsize[:2])),
                    p_sault=ka[i]*sault_p, p_pepper=ka[i]*pepper_p)
        else:
            simu_img = pepper_sault(ka * beta_c_array(sigma_beta, size=imgsize) * (
                shot_noise(ne_array)
                + dark_current(nd_factor * exp_time, size=imgsize)
                + read_noise(sigma_r, size=imgsize)),
                p_sault=ka*sault_p, p_pepper=ka*pepper_p)

        simu_img = simu_img.astype(int) * kd
        # Quantify and Digital Gain
        # kc = ka * kd

        simu_img = (np.clip(simu_img, 0, 255) / 255) ** gamma * 255
        simu_img = np.clip(simu_img, 0, 255).astype(int)
        return simu_img

    def take_photo_M(self, img_source=None, exp_time=1, iso=100, aperture=2,
                     illumination_factor=1, imgsize=None, ka=2, kd=3):

        if imgsize is None:
            imgsize = img_source.shape
        img = self.simu_noise(img_source=img_source,
                              sigma_beta=self.sigma_beta,
                              nd_factor=self.nd_factor,
                              sigma_r=self.sigma_r,
                              ka=(np.array(ka)).tolist(),
                              exp_time=exp_time,
                              illumination=(1 / aperture ** 2) *
                              illumination_factor,
                              imgsize=imgsize,
                              kd=kd)

        img = np.clip(img, 0, 255)

        return img.astype(np.uint8)

    def take_photo_P(self, img_source=None, imgsize=None, ka=1, kd=3):

        if imgsize is None:
            imgsize = img_source.shape
        # print(self.sigma_beta, self.sigma_r, self.nd_factor, ka, kd)
        img = self.simu_noise(img_source=img_source,
                              sigma_beta=self.sigma_beta,
                              nd_factor=self.nd_factor,
                              sigma_r=self.sigma_r,
                              ka=(np.array(ka)).tolist(),
                              exp_time=1,
                              illumination=(1/np.array(ka*kd)).tolist(),
                              imgsize=imgsize,
                              kd=kd)

        img = np.clip(img, 0, 255)

        return img.astype(np.uint8)

# ------------------
# main class
# ------------------


class BasicNoiseModel(object):
    """
    Basic Noise models (numpy based) including gaussian, poisson, sault-pepper
    """
    # NotImplementment Now

    def __init__(self, noise_type: str, noise_parameters: dict = {}):
        """
        add noise to clear image

        Args:
            noise_model (str): noise model type, 'gaussian'|'camera'
            noise_params (dict): noise model parameters (fixed params for noise model)
        Returns:
            np.ndarray: noisy image
        """
        self.noise_model = noise_type
        self.noise_parameters = noise_parameters

    def gaussian_noise(self, clear_img):
        # awgn - gaussian noise model
        # clear_img (np.ndarray): clear image
        # sigma (scalar|list-range): guassian noise level
        sigma = self.noise_parameters['sigma']
        # uniform random pick param
        if isinstance(sigma, (int, float)):
            noise_level = sigma
        else:
            noise_level = np.random.uniform(*sigma)
        # add noise
        img_noisy = clear_img + \
            np.random.normal(0, noise_level, clear_img.shape)
        return img_noisy, {'sigma': noise_level}

    def add_noise(self, clear_img):
        # add noise with assigned noise model and noise params
        # clear_img (np.ndarray): clear image
        if self.noise_model.lower() == 'gaussian':
            return self.gaussian_noise(clear_img)
