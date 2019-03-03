import numpy as np
import cv2


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
            image = cv2.drawContours(image, [contour], -1, (0, 0, 0), -1)
        elif area <= 7500:
            image = cv2.drawContours(image, [contour], -1, (255, 0, 0), -1)
        elif 37500 > area >= 7500:
            image = cv2.drawContours(image, [contour], -1, (0, 255, 0), -1)
        elif area >= 37500:
            image = cv2.drawContours(image, [contour], -1, (0, 0, 255), -1)

    return image
