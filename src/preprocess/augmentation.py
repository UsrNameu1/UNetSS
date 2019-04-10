from pathlib import Path
import warnings

from cv2 import imread
from skimage.io import imsave
import cv2
from albumentations import (
    RandomCrop,
    Compose,
    Transpose,
    RandomRotate90,
    OneOf,
    RGBShift,
    HueSaturationValue
)

from preprocess.labeling import transform_to_sizelabel
import settings


__all__ = ['random_crop', 'runtime_augmentor']


def random_crop(
    image_dir: Path, gt_dir: Path,
    filename: str, sample_size: int, crop_size: int, use_size_label: bool, output_dir: Path
):
    """
    randomly crop fixed size image sample to train model
    :param image_dir: image directory
    :param gt_dir: gt label directory
    :param filename: filename to image
    :param sample_size: sampling size per image
    :param crop_size: clopping image size
    :param use_size_label: if True, use size specific label gt output
    :param output_dir: output directory
    """
    image_path = image_dir.joinpath(filename)
    gt_path = gt_dir.joinpath(filename)

    image = imread(image_path.as_posix(), cv2.IMREAD_COLOR)

    if not use_size_label:
        gt_image = imread(gt_path.as_posix(), cv2.IMREAD_GRAYSCALE)
    else:
        gt_image = imread(gt_path.as_posix(), cv2.IMREAD_GRAYSCALE)
        gt_image = transform_to_sizelabel(gt_image)

    aug = RandomCrop(width=crop_size, height=crop_size)
    file_stem = filename.split('.')[0]

    for i in range(sample_size):
        augmented = aug(image=image, mask=gt_image)
        image_cropped = augmented['image']
        gt_cropped = augmented['mask']
        imsave(output_dir.joinpath(settings.image_subdir_name, settings.dummycls_name,
                                   '{}_{}.png'.format(file_stem, i)).as_posix(), image_cropped)
        imsave(output_dir.joinpath(settings.gt_subdir_name, settings.dummycls_name,
                                   '{}_{}.png'.format(file_stem, i)).as_posix(), gt_cropped, check_contrast=False)



"""
augmentation pipeline for runtime
"""
runtime_augmentor = Compose([
    OneOf([
        Transpose(p=0.5),
        RandomRotate90(p=0.5)
    ], p=0.5),
    OneOf([
        RGBShift(r_shift_limit=10, g_shift_limit=10, b_shift_limit=10),
        HueSaturationValue(hue_shift_limit=10, sat_shift_limit=15, val_shift_limit=10)
    ], p=0.3)
])
