from typing import Tuple

from keras.models import Model
from keras.layers import (
    Input,
    Conv2D,
    MaxPool2D,
    UpSampling2D,
    concatenate,
    BatchNormalization,
    Activation
)


def conv_block(input_tensor, filters: int, kernel_size: int = 3, iterations: int = 2, batch_normalize: bool = False):
    x = input_tensor
    for _ in range(iterations):
        x = Conv2D(filters, kernel_size, padding='same', kernel_initializer='he_normal')(x)
        x = BatchNormalization()(x) if batch_normalize else x
        x = Activation('relu')(x)
    return x


def upconv_block(input_tensor, filters: int, kernel_size: int = 2):
    x = UpSampling2D(size=(2, 2))(input_tensor)
    x = Conv2D(filters, kernel_size, activation='relu', padding='same', kernel_initializer='he_normal')(x)
    return x


def UNet(weights: str = None,
         input_shape: Tuple[int, int, int] = (None, None, 3),
         batch_normalize: bool = False,
         last_channels: int = 2) -> Model:
    """Instantiates the U-Net architecture

    :param weights: the path to the weights file to be loaded
    :param input_shape: optional shape tuple, only to be specified
    :param batch_normalize: if True, apply batch normalization after each convolution layer except upconv block
    :param last_channels: optional number of channels of segmentation output mapping image
    :return: A Keras model instance
    """

    # Encoding Layers
    inputs = Input(input_shape)
    conv1 = conv_block(inputs, 64, batch_normalize)
    pool1 = MaxPool2D(pool_size=(2, 2))(conv1)
    conv2 = conv_block(pool1, 128, batch_normalize)
    pool2 = MaxPool2D(pool_size=(2, 2))(conv2)
    conv3 = conv_block(pool2, 256, batch_normalize)
    pool3 = MaxPool2D(pool_size=(2, 2))(conv3)
    conv4 = conv_block(pool3, 512, batch_normalize)
    pool4 = MaxPool2D(pool_size=(2, 2))(conv4)
    conv5 = conv_block(pool4, 1024, batch_normalize)

    # Decoding Layers
    upconv6 = upconv_block(conv5, 512)
    merge6 = concatenate([conv4, upconv6], axis=3)
    conv6 = conv_block(merge6, 512, batch_normalize)
    upconv7 = upconv_block(conv6, 256)
    merge7 = concatenate([conv3, upconv7], axis=3)
    conv7 = conv_block(merge7, 256, batch_normalize)
    upconv8 = upconv_block(conv7, 128)
    merge8 = concatenate([conv2, upconv8], axis=3)
    conv8 = conv_block(merge8, 128, batch_normalize)
    upconv9 = upconv_block(conv8, 64)
    merge9 = concatenate([conv1, upconv9], axis=3)
    conv9 = conv_block(merge9, 64, batch_normalize)
    output = Conv2D(last_channels, 1,
                    activation='softmax',
                    padding='same',
                    kernel_initializer='he_normal')(conv9)

    model = Model(inputs, output)

    if weights is not None:
        model.load_weights(weights)

    return model
