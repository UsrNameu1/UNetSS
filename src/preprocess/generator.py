from pathlib import Path
from typing import Union, Generator

import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.np_utils import to_categorical
from albumentations import (
    Compose,
    Transpose,
    RandomRotate90,
    OneOf,
    RGBShift,
    HueSaturationValue
)

import settings
from settings import BinLabel, SizeLabel
from preprocess.labeling import to_cls_map


def data_generator(root_path: Path, batch_size: int, label: Union[BinLabel, SizeLabel]) -> Generator:
    """
    generator for randomly cropped images
    :param root_path: root directory for input images generated with endpoints/augment_traindata.py
    :param batch_size: batch size
    :param label: label settings
    :return:
    """
    data_gen_args = dict(horizontal_flip=True, vertical_flip=True)
    image_datagen = ImageDataGenerator(**data_gen_args)
    gt_datagen = ImageDataGenerator(**data_gen_args)

    seed = 31
    image_generator = image_datagen.flow_from_directory(
        root_path.joinpath(settings.image_subdir_name).as_posix(),
        batch_size=batch_size,
        class_mode=None,
        seed=seed
    )
    gt_generator = gt_datagen.flow_from_directory(
        root_path.joinpath(settings.gt_subdir_name).as_posix(),
        batch_size=batch_size,
        class_mode=None,
        seed=seed
    )

    ret_generator = (_aug_transform(img_batch, gt_batch, label)
                     for img_batch, gt_batch in zip(image_generator, gt_generator))
    return ret_generator


_augmentor = Compose([
    OneOf([
        Transpose(p=0.5),
        RandomRotate90(p=0.5)
    ], p=0.5),
    OneOf([
        RGBShift(r_shift_limit=10, g_shift_limit=10, b_shift_limit=10),
        HueSaturationValue(hue_shift_limit=10, sat_shift_limit=15, val_shift_limit=10)
    ], p=0.3)
])


def _aug_transform(
    img_batch: np.ndarray, gt_batch: np.ndarray, label: Union[BinLabel, SizeLabel]
) -> (np.ndarray, np.ndarray):
    batch_auged = (_augmentor(image=img, mask=gt) for img, gt in zip(img_batch, gt_batch))
    imgs_auged, gts_auged = zip(
        *((auged['image'] / 255.,
           to_categorical(to_cls_map(auged['mask'].astype(np.uint8), label), num_classes=len(label)))
          for auged in batch_auged)
    )
    return np.stack(imgs_auged), np.stack(gts_auged)
