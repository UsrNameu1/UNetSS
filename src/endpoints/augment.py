from pathlib import Path
import math
from itertools import product

from sklearn.model_selection import train_test_split
from joblib import Parallel, delayed
import click
from loguru import logger

from preprocess.augmentation import random_crop
from settings import (
    train_subdir_name,
    validation_subdir_name,
    image_subdir_name,
    gt_subdir_name,
    dummycls_name
)
from settings.logging import configure_logging

configure_logging()


@click.command()
@click.option('--image_dir', type=click.Path(exists=True), required=True, help="aerial image directory")
@click.option('--gt_dir', type=click.Path(exists=True), required=True, help='ground truth image directory')
@click.option("--sample_size", type=click.INT, required=True, help='sample size')
@click.option("--output_dir", type=click.Path(exists=False), required=True, help='output directory')
@click.option("--crop_size", type=click.INT, required=False, default=256, help='clopping size')
@click.option("--validation_ratio", type=click.FloatRange(min=0.0, max=0.3), required=False, default=0.0,
              help='ratio of validation files')
@click.option("--resize_ratio", type=click.FloatRange(min=0.0, max=1.0), required=False, default=1.0,
              help='resize output ratio')
@click.option("--use_size_label", is_flag=True, required=False, default=False, help='apply size specific label')

def augment(
    image_dir: str, gt_dir: str, sample_size: int, output_dir: str, crop_size: int, validation_ratio: float,
    resize_ratio: float, use_size_label: bool
):
    image_dir = Path(image_dir)
    file_names = list(path.name for path in Path(image_dir).glob('./*'))
    train_file_names, validation_file_names = train_test_split(file_names, test_size=validation_ratio)
    gt_dir = Path(gt_dir)

    image_count = len(file_names)
    sample_size_per_image = math.ceil(sample_size / image_count)

    output_dir = Path(output_dir)

    if not output_dir.exists():
        for subdir, subsubdir in product([train_subdir_name, validation_subdir_name],
                                         [image_subdir_name, gt_subdir_name]):
            output_dir.joinpath(subdir, subsubdir, dummycls_name).mkdir(parents=True)

    logger.debug("train sample count: {}".format(len(train_file_names)))
    Parallel(n_jobs=-1)(delayed(random_crop)(
        image_dir=image_dir, gt_dir=gt_dir, filename=name, sample_size=sample_size_per_image, crop_size=crop_size,
        resize_ratio=resize_ratio, use_size_label=use_size_label, output_dir=output_dir.joinpath(train_subdir_name)
    ) for name in train_file_names)
    logger.debug("validation sample count: {}".format(len(validation_file_names)))
    Parallel(n_jobs=-1)(delayed(random_crop)(
        image_dir=image_dir, gt_dir=gt_dir, filename=name, sample_size=sample_size_per_image, crop_size=crop_size,
        resize_ratio=resize_ratio, use_size_label=use_size_label, output_dir=output_dir.joinpath(validation_subdir_name)
    ) for name in validation_file_names)
    logger.debug("done")


if __name__ == '__main__':
    augment()
