from pathlib import Path
import math
import json

from keras import optimizers
from keras.callbacks import CSVLogger, ReduceLROnPlateau, ModelCheckpoint
import click

from applications import UNet, UNetSS
from preprocess.generator import data_generator
from settings import SizeLabel, BinLabel
from settings.logging import configure_logging
import settings

configure_logging()


@click.command()
@click.option('--config_path', type=click.Path(exists=True), required=True, help="configuration json path")
def train(config_path: str):
    config_file = open(config_path)
    configs = json.load(config_file)

    model_config = configs["model"]

    if model_config["architecture"] == "UNet":
        model = UNet(batch_normalize=model_config["batch_normalize"])
    elif model_config["architecture"] == "UNetSS":
        model = UNetSS(batch_normalize=model_config["batch_normalize"])
    else:
        raise ValueError("unknown segmentation architecture")

    optimizer_config = dict()
    optimizer_config['class_name'] = configs["optimizer"]["name"]
    optimizer_config['config'] = configs["optimizer"]["args"]
    optimizer = optimizers.deserialize(optimizer_config)

    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    train_root_dir = Path(configs["train_data"]["root_dir"], settings.train_subdir_name)
    validation_root_dir = Path(configs["train_data"]["root_dir"], settings.validation_subdir_name)
    batch_size = configs["train_data"]["batch_size"]
    label = SizeLabel if configs["train_data"]["use_size_label"] else BinLabel
    train_output_branched = configs["train_data"]["branched"]
    train_data_generator = data_generator(train_root_dir, batch_size, label, train_output_branched)
    validation_data_generator = data_generator(validation_root_dir, batch_size, label, train_output_branched)

    train_sample_n = \
        len(list(train_root_dir.joinpath(settings.image_subdir_name, settings.dummycls_name).glob('./*.png')))
    validation_sample_n = \
        len(list(validation_root_dir.joinpath(settings.image_subdir_name, settings.dummycls_name).glob('./*.png')))
    epochs = configs["train_data"]["epochs"]

    output_dir = Path(configs["output_dir"])
    if not output_dir.exists():
        output_dir.mkdir(parents=True)

    callbacks = [
        CSVLogger(output_dir.joinpath('training.csv'), append=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.00001),
        ModelCheckpoint(output_dir.joinpath('weights_{epoch}.h5').as_posix(), monitor='val_loss', save_best_only=True,
                        save_weights_only=True, mode='auto', period=5)
    ]

    model.fit_generator(
        train_data_generator, int(math.ceil(train_sample_n / batch_size)),
        callbacks=callbacks,
        validation_data=validation_data_generator, validation_steps=validation_sample_n,
        epochs=epochs
    )

    model.save_weights(output_dir.joinpath('weights_{}.h5'.format(epochs)).as_posix())


if __name__ == '__main__':
    train()
