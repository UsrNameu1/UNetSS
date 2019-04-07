# Endpoint execution examples


## augment from inria dataset

### binary label

```bash
docker exec -itd unetss python3 src/endpoints/augment.py \
--image_dir input/AerialImageDataset/train/images/ \
--gt_dir input/AerialImageDataset/train/gt/ \
--sample_size 10000 --crop_size 256 --validation_ratio 0.05 \
--output_dir input/train_bin_256
```

### size specific label

```bash
docker exec -itd unetss python3 src/endpoints/augment.py \
--image_dir input/AerialImageDataset/train/images/ \
--gt_dir input/AerialImageDataset/train/gt/ \
--sample_size 10000 --crop_size 256 --validation_ratio 0.05 \
--output_dir input/train_sizelabel_256 --use_size_label
```

## train model

### unet architecture

```bash
docker exec -itd unetss python3 src/endpoints/train.py \
--config_path configs/train/train_unet.json 
```

### size specific unet architecture 

```bash
docker exec -itd unetss python3 src/endpoints/train.py \
--config_path configs/train/train_unetss.json
```
