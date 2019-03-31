# Endpoint execution examples


## augment from inria dataset

```bash
docker exec -itd unetss bash -c \
"cd /tf/scripts && python3 src/endpoints/augment.py \
--image_dir input/AerialImageDataset/train/images/ \
--gt_dir input/AerialImageDataset/train/gt/ \
--sample_size 10000 --crop_size 256 \
--output_dir input/train_bin_256 &>> out.log"
```

## train unet

```bash
docker exec -itd unetss bash -c \
"cd /tf/scripts && python3 src/endpoints/train.py \
--config_path configs/train/train_unet.json &>> out.log"
```

## train unet with size specific tasks

```bash
docker exec -itd unetss bash -c \
"cd /tf/scripts && python3 src/endpoints/train.py \
--config_path configs/train/train_unetss.json &>> out.log"
```
