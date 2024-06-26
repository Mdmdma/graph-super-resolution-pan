from pathlib import Path
import argparse
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms import InterpolationMode, Resize
import torch.nn.functional as F
import os
import cv2
import gc
import matplotlib.pyplot as plt

from .utils import downsample, random_crop, random_rotate, random_horizontal_flip

Pan_BASE_SIZE = (1000, 1000)


class PanDataset(Dataset):

    def __init__(
            self,
            data_dir: str,
            resolution='HR',
            scale=1.0,
            crop_size=(256, 256),
            do_horizontal_flip=True,
            max_rotation_angle: int = 15,
            scale_interpolation=InterpolationMode.BILINEAR,
            rotation_interpolation=InterpolationMode.BILINEAR,
            image_transform=None,
            in_memory=True,
            split='train',
            crop_valid=False,
            crop_deterministic=False,
            scaling=8,
    ):
        self.scale = scale
        self.crop_size = crop_size
        self.do_horizontal_flip = do_horizontal_flip
        self.max_rotation_angle = max_rotation_angle
        self.scale_interpolation = scale_interpolation
        self.rotation_interpolation = rotation_interpolation
        self.image_transform = image_transform
        self.crop_valid = crop_valid
        self.crop_deterministic = crop_deterministic
        self.scaling = scaling
        data_dir = Path(data_dir)

        if max_rotation_angle > 0 and crop_deterministic:
            raise ValueError('Max rotation angle has to be zero when cropping deterministically')

        if split not in ('train', 'val', 'test'):
            raise ValueError(split)

        mmap_mode = None if in_memory else 'c'

        # Select one of the lines below to load the images in bulk or one by one
        #self.images = [os.path.join(data_dir, split, f) for f in os.listdir(os.path.join(data_dir, split)) if f.endswith('.tif')]
        self.images = np.load(str(data_dir / f'npy/images_{split}.npy'), mmap_mode)

        self.H, self.W = int(Pan_BASE_SIZE[0] * self.scale), int(Pan_BASE_SIZE[1] * self.scale)

        if self.crop_valid:
            if self.max_rotation_angle > 45:
                raise ValueError('When crop_valid=True, only rotation angles up to 45° are supported for now')

            # make sure that max rotation angle is valid, else decrease
            max_angle = np.floor(min(
                2 * np.arctan
                    ((np.sqrt(-(crop_size[0] ** 2) + self.H ** 2 + self.W ** 2) - self.W) / (crop_size[0] + self.H)),
                2 * np.arctan
                    ((np.sqrt(-(crop_size[1] ** 2) + self.W ** 2 + self.H ** 2) - self.H) / (crop_size[1] + self.W))
            ) * (180. / np.pi))

            if self.max_rotation_angle > max_angle:
                print(f'max rotation angle too large for given image size and crop size, decreased to {max_angle}')
                self.max_rotation_angle = max_angle

    def __getitem__(self, index):
        if self.crop_deterministic:
            num_crops_h, num_crops_w = self.H // self.crop_size[0], self.W // self.crop_size[1]
            im_index = index // (num_crops_h * num_crops_w)
        else:
            im_index = index

        image = torch.from_numpy(self.images[im_index].astype('float32')) / 255.

        resize = Resize((self.H, self.W), self.scale_interpolation)
        image = resize(image)

        if self.do_horizontal_flip and not self.crop_deterministic:
            image = random_horizontal_flip(image)

        if self.max_rotation_angle > 0  and not self.crop_deterministic:
            image = random_rotate(image, self.max_rotation_angle, self.rotation_interpolation,
                    crop_valid=self.crop_valid)

        if self.crop_deterministic:
            crop_index = index % (num_crops_h * num_crops_w)
            crop_index_h, crop_index_w = crop_index // num_crops_w, crop_index % num_crops_w
            slice_h = slice(crop_index_h * self.crop_size[0], (crop_index_h + 1) * self.crop_size[0])
            slice_w = slice(crop_index_w * self.crop_size[1], (crop_index_w + 1) * self.crop_size[1])
            image = image[:, slice_h, slice_w]
        else:
            image = random_crop(image, self.crop_size)

        if self.image_transform is not None:
            image = self.image_transform(image)

        bw_image = torch.mean(image, 0, keepdim=True)
        source = downsample(image.unsqueeze(0), self.scaling).squeeze()

        mask_hr = (~torch.isnan(image)).float()
        mask_lr = (~torch.isnan(source)).float()

        source[mask_lr == 0.] = 0.

        y_bicubic = F.interpolate(source.unsqueeze(0), scale_factor=self.scaling, mode='bicubic', align_corners=False).float()
        y_bicubic = y_bicubic.reshape((3, self.crop_size[0], self.crop_size[1]))
        return {'guide': bw_image, 'y': image, 'source': source, 'mask_hr': mask_hr, 'mask_lr': mask_lr, 
            'y_bicubic': y_bicubic}

    def __len__(self):
        if self.crop_deterministic:
            return len(self.images) * (self.H // self.crop_size[0]) * (self.W // self.crop_size[1])
        return len(self.images)