import re
import csv
import random
import warnings

import numpy as np
import torch
from torchvision.transforms import RandomCrop, RandomRotation
import torchvision.transforms.functional as F
from skimage.measure import block_reduce
from scipy import interpolate
from scipy.ndimage import zoom

ROTATION_EXPAND = False
ROTATION_CENTER = None  # image center
ROTATION_FILL = 0.


def downsample(image, scaling_factor):
    """
    Performs average pooling, ignoring nan values
    :param image: torch tensor or numpy ndarray of shape (B, C, H, W)
    """
    if image.ndim != 4:
        raise ValueError(f'Image should have four dimensions, got {image.ndim}')

    is_tensor = torch.is_tensor(image)
    if is_tensor:
        device = image.device
        image = image.detach().cpu().numpy()

    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', r'Mean of empty slice')
        image = block_reduce(image, (1, 1, scaling_factor, scaling_factor), np.nanmean)

    return torch.from_numpy(image).to(device) if is_tensor else image


def random_horizontal_flip(image, p=0.5):
    if random.random() < p:
        return image.flip(-1)
    return image


def random_rotate(image, max_rotation_angle, interpolation, crop_valid=False):
    angle = RandomRotation.get_params([-max_rotation_angle, max_rotation_angle])
    if crop_valid:
        rotated = F.rotate(image, angle, interpolation, True, ROTATION_CENTER, ROTATION_FILL)
        crop_params = np.floor(np.asarray(rotated.shape[1:3]) - 2. *(np.sin(np.abs(angle * np.pi / 180.)) * np.asarray(image.shape[1:3][::-1]))).astype(int)
        return F.center_crop(image, crop_params)
    else:
        return F.rotate(image, angle, interpolation, ROTATION_EXPAND, ROTATION_CENTER, ROTATION_FILL)


def random_crop(image, crop_size):
    crop_params = RandomCrop.get_params(image, crop_size)
    return F.crop(image, *crop_params)


# Following contents were adapted from https://www.programmersought.com/article/2506939342/.


def _read_pfm(pfm_file_path):
    with open(pfm_file_path, 'rb') as pfm_file:
        header = pfm_file.readline().decode('utf-8').rstrip()
        channels = 3 if header == 'PF' else 1

        dim_match = re.match(r'^(\d+)\s(\d+)\s$', pfm_file.readline().decode('utf-8'))
        if dim_match:
            width, height = map(int, dim_match.groups())
        else:
            raise Exception('Malformed PFM header.')

        scale = float(pfm_file.readline().decode('utf-8').rstrip())
        if scale < 0:
            endian = '<'  # little endian
        else:
            endian = '>'  # big endian

        disparity = np.fromfile(pfm_file, endian + 'f')

    return disparity, (height, width, channels)


def read_calibration(calib_file_path):
    with open(calib_file_path, 'r') as calib_file:
        calib = {}
        csv_reader = csv.reader(calib_file, delimiter='=')
        for attr, value in csv_reader:
            calib.setdefault(attr, value)

    return calib


def create_depth_from_pfm(pfm_file_path, calib=None):
    disparity, shape = _read_pfm(pfm_file_path)

    if calib is None:
        raise Exception('No calibration information available')
    else:
        fx = float(calib['cam0'].split(' ')[0].lstrip('['))
        base_line = float(calib['baseline'])
        doffs = float(calib['doffs'])

        depth_map = fx * base_line / (disparity + doffs)
        depth_map = np.reshape(depth_map, newshape=shape)
        depth_map = np.flipud(depth_map).transpose((2, 0, 1)).copy()

        depth_map[depth_map == 0.] = np.nan

        return depth_map
