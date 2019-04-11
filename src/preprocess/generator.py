from pathlib import Path
from typing import Union, Generator, List

import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.np_utils import to_categorical

from .augmentation import runtime_augmentor
import settings
from settings import BinLabel, SizeLabel
from preprocess.labeling import to_cls_map, to_cls_branch_maps


__all__ = ['data_generator']


def data_generator(root_path: Path,
                   batch_size: int,
                   label: Union[BinLabel, SizeLabel],
                   branched: bool = False) -> Generator:
    """
    generator for randomly cropped images
    :param root_path: root directory for input images generated with endpoints/augment.py
    :param batch_size: batch size
    :param label: label settings
    :param branched: if True, output tensor is one hot segmentation map of existence on each label
    :return:
    """
    data_gen_args = dict(horizontal_flip=True, vertical_flip=True)
    image_datagen = ImageDataGenerator(**data_gen_args)
    gt_datagen = ImageDataGenerator(**data_gen_args)

    seed = 37
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

    ret_generator = (_aug_transform(img_batch, gt_batch, label, branched)
                     for img_batch, gt_batch in zip(image_generator, gt_generator))
    return ret_generator


def _aug_transform(
    img_batch: np.ndarray, gt_batch: np.ndarray, label: Union[BinLabel, SizeLabel], branched: bool = False
) -> (np.ndarray, Union[np.ndarray, List[np.ndarray]]):
    batch_auged = (runtime_augmentor(image=img.astype(np.uint8), mask=gt) for img, gt in zip(img_batch, gt_batch))
    imgs_auged, gts_auged = zip(
        *([auged['image'] / 255., auged['mask'].astype(np.uint8)]
          for auged in batch_auged)
    )
    if branched:
        gts_auged_list = zip(
            *([to_categorical(branch_map, num_classes=2) for branch_map in to_cls_branch_maps(gt_auged, label)]
              for gt_auged in gts_auged)
        )
        return np.stack(imgs_auged), [np.stack(gts_auged) for gts_auged in gts_auged_list]
    else:
        gts_auged = [to_categorical(to_cls_map(gt_auged, label), num_classes=len(label)) for gt_auged in gts_auged]
        return np.stack(imgs_auged), np.stack(gts_auged)
