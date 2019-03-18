from pathlib import Path
import math

from cv2 import imread, imwrite
import cv2
from joblib import Parallel, delayed
import click
from loguru import logger
from albumentations import RandomCrop

from preprocess.labeling import transform_to_sizelabel
import settings


@click.command()
@click.option('--image_dir', type=click.Path(exists=True), required=True, help="aerial image directory")
@click.option('--gt_dir', type=click.Path(exists=True), required=True, help='ground truth image directory')
@click.option("--sample_size", type=int, required=True, help='sample size')
@click.option("--output_dir", type=click.Path(exists=False), required=True, help='output directory')
@click.option("--crop_size", type=int, required=False, default=256, help='clopping size')
@click.option("--use_size_label", is_flag=True, required=False, default=False, help='apply size specific label')
def augment_traindata(
    image_dir: str, gt_dir: str, sample_size: int, output_dir: str, crop_size: int, use_size_label: bool
):
    image_dir = Path(image_dir)
    file_names = list(path.name for path in Path(image_dir).glob('./*'))
    gt_dir = Path(gt_dir)

    image_count = len(file_names)
    sample_size_per_image = math.ceil(sample_size / image_count)

    output_dir = Path(output_dir)

    if not output_dir.exists():
        output_dir.joinpath(settings.image_subdir_name, settings.dummycls_name).mkdir(parents=True)
        output_dir.joinpath(settings.gt_subdir_name, settings.dummycls_name).mkdir(parents=True)

    logger.debug("sample count: {}".format(image_count))
    Parallel(n_jobs=-1)(delayed(_random_crop)(
        image_dir=image_dir, gt_dir=gt_dir, filename=name, sample_size=sample_size_per_image, crop_size=crop_size,
        use_size_label=use_size_label, output_dir=output_dir
    ) for name in file_names)
    logger.debug("done")


def _random_crop(
    image_dir: Path, gt_dir: Path, filename: str, sample_size: int, crop_size: int,
    use_size_label: bool, output_dir: Path
):
    image_path = image_dir.joinpath(filename)
    gt_path = gt_dir.joinpath(filename)

    image = imread(image_path.as_posix(), cv2.IMREAD_COLOR)

    if not use_size_label:
        gt_image = imread(gt_path.as_posix(), cv2.IMREAD_GRAYSCALE)
    else:
        gt_image = imread(gt_path.as_posix(), cv2.IMREAD_GRAYSCALE)
        gt_image = transform_to_sizelabel(gt_image)
        gt_image = cv2.cvtColor(gt_image, cv2.COLOR_RGB2BGR)

    aug = RandomCrop(width=crop_size, height=crop_size)
    file_stem = filename.split('.')[0]

    for i in range(sample_size):
        augmented = aug(image=image, mask=gt_image)
        image_cropped = augmented['image']
        gt_cropped = augmented['mask']
        imwrite(output_dir.joinpath(settings.image_subdir_name, settings.dummycls_name,
                                    '{}_{}.png'.format(file_stem, i)).as_posix(), image_cropped)
        imwrite(output_dir.joinpath(settings.gt_subdir_name, settings.dummycls_name,
                                    '{}_{}.png'.format(file_stem, i)).as_posix(), gt_cropped)


if __name__ == '__main__':
    augment_traindata()
