# Dataset

Training and evaluation are executed on this dataset

[Emmanuel Maggiori, Yuliya Tarabalka, Guillaume Charpiat and Pierre Alliez. “Can Semantic Labeling Methods Generalize to Any City? The Inria Aerial Image Labeling Benchmark”. IEEE International Geoscience and Remote Sensing Symposium (IGARSS). 2017.](https://project.inria.fr/aerialimagelabeling/)

## Download input images

```bash
mkdir input
cd input
curl -k https://files.inria.fr/aerialimagelabeling/getAerial.sh | bash
```

## Extract chicago data as test data

```bash
cd input/AerialImageDataset
mv test test_nolabel
mkdir -p test/{images,gt}
mv train/images/chicago* test/images/
mv train/gt/chicago* test/gt/
```
