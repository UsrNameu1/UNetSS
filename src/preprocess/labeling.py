from typing import List
from enum import Enum

import numpy as np
import cv2

from settings.label import SizeLabel


def transform_to_sizelabel(bin_image: np.ndarray) -> np.ndarray:
    """
    transform binary segmentation label to three size specific label
    :param bin_image: binary segmentation label
    :return: three size specific label (r: small, g: medium, b: large)
    """
    contours, hierarchy = cv2.findContours(
        bin_image,
        cv2.RETR_TREE,
        cv2.CHAIN_APPROX_SIMPLE
    )

    height, width = bin_image.shape[0], bin_image.shape[1]
    image = np.zeros((height, width, 3), dtype=np.uint8)

    for idx, contour in enumerate(contours):
        area = cv2.contourArea(contour)

        parent_idx = hierarchy[0][idx][3]
        if parent_idx >= 0:
            image = cv2.drawContours(image, [contour], -1, SizeLabel.blank.value.rgb, -1)
        elif area <= 3400:
            image = cv2.drawContours(image, [contour], -1, SizeLabel.small.value.rgb, -1)
        elif 18800 > area >= 3400:
            image = cv2.drawContours(image, [contour], -1, SizeLabel.medium.value.rgb, -1)
        elif area >= 18800:
            image = cv2.drawContours(image, [contour], -1, SizeLabel.large.value.rgb, -1)

    return image


def to_cls_map(label_img: np.ndarray, label: Enum) -> np.ndarray:
    """
    transform gt label image to class map
    :param label_img: gt label image
    :param label: labeling information enum (see settings)
    :return: class map image
    """
    w, h, _ = label_img.shape
    cls_map = np.zeros((w, h), dtype=np.uint8)
    for cls in label:
        cls_map[np.all(label_img == cls.value.rgb, axis=-1)] = cls.value.cls
    return cls_map
