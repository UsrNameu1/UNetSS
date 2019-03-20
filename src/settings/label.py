from enum import Enum

import numpy as np


class ColorWithCls:
    """
    Class information with rgb color values
    """

    def __init__(
        self, cls: int, rgb: (np.uint8, np.uint8, np.uint8), color: str
    ):
        self.cls = cls
        self.rgb = rgb
        self.color = color


class BinLabel(Enum):
    """
    Binary labels for building footprints
    """
    building = ColorWithCls(1, (0xFF, 0xFF, 0xFF), 'white')
    blank = ColorWithCls(0, (0x00, 0x00, 0x00), 'black')


class SizeLabel(Enum):
    """
    Size specific labels for building footprints
    """
    small = ColorWithCls(1, (0xFF, 0x00, 0x00), 'red')
    medium = ColorWithCls(2, (0x00, 0xFF, 0x00), 'green')
    large = ColorWithCls(3, (0x00, 0x00, 0xFF), 'blue')
    blank = ColorWithCls(0, (0x00, 0x00, 0x00), 'black')
