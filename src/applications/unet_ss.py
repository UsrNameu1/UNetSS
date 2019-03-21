from typing import Tuple

from keras.models import Model
from keras.layers import (
    Input,
    Conv2D,
    MaxPool2D,
    concatenate
)

from .unet import conv_block, upconv_block


def UNetSS(weights: str = None,
           input_shape: Tuple[int, int, int] = (None, None, 3),
           batch_normalize: bool = False,
           last_branches: int = 3) -> Model:
    """Instantiates the U-Net architecture for Size Specific tasks

    :param weights: the path to the weights file to be loaded
    :param input_shape: optional shape tuple, only to be specified
    :param batch_normalize: if True, apply batch normalization after each convolution layer except upconv block
    :param last_branches: optional number of branches of task by size specific labels
    :return: A Keras model instance
    """

    # Encoding Layers
    inputs = Input(input_shape)
    conv1 = conv_block(inputs, 64, batch_normalize=batch_normalize)
    pool1 = MaxPool2D(pool_size=(2, 2))(conv1)
    conv2 = conv_block(pool1, 128, batch_normalize=batch_normalize)
    pool2 = MaxPool2D(pool_size=(2, 2))(conv2)
    conv3 = conv_block(pool2, 256, iterations=3, batch_normalize=batch_normalize)
    pool3 = MaxPool2D(pool_size=(2, 2))(conv3)
    conv4 = conv_block(pool3, 512, iterations=3, batch_normalize=batch_normalize)
    pool4 = MaxPool2D(pool_size=(2, 2))(conv4)
    conv5 = conv_block(pool4, 1024, iterations=4, batch_normalize=batch_normalize)

    # Decoding Layers
    upconv6 = upconv_block(conv5, 512)
    merge6 = concatenate([conv4, upconv6], axis=3)
    conv6 = conv_block(merge6, 512, batch_normalize=batch_normalize)
    upconv7 = upconv_block(conv6, 256)
    merge7 = concatenate([conv3, upconv7], axis=3)
    conv7 = conv_block(merge7, 256, batch_normalize=batch_normalize)
    upconv8 = upconv_block(conv7, 128)

    # Size specific branch decoding
    outputs = []
    for i in range(last_branches):
        merge8 = concatenate([conv2, upconv8], axis=3)
        conv8 = conv_block(merge8, 128, iterations=1, batch_normalize=batch_normalize)
        upconv9 = upconv_block(conv8, 64)
        merge9 = concatenate([conv1, upconv9], axis=3)
        conv9 = conv_block(merge9, 64, iterations=1, batch_normalize=batch_normalize)
        output = Conv2D(2, 1,
                        activation='softmax',
                        padding='same',
                        kernel_initializer='he_normal',
                        name='o{}'.format(i + 1))(conv9)
        outputs.append(output)

    model = Model(inputs, outputs)

    if weights is not None:
        model.load_weights(weights)

    return model
