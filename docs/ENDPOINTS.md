# Endpoint execution examples


## augment from inria dataset

```bash
docker exec -it unetss sh -c \
    "cd /tf/scripts && python3 src/endpoints/augment.py \"
    "--image_dir input/AerialImageDataset/train/images/ \"
    "--gt_dir input/AerialImageDataset/train/gt/ \"
    "--sample_size 10000 --crop_size 512 \"
    "--output_dir input/train_bin_512 >> out.log"
```

## train unet+ss

```
docker exec -it unetss sh -c \
    "cd /tf/scripts && python3 src/endpoints/train.py \"
    "--config_path src/endpoints/configs/train/train_unetss.json > out.log"
```
