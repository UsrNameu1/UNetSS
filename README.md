Size specific UNet
====

An implementation and evaluation of the following paper (road distillation channel excluded).

[Ryuhei Hamaguchi and Shuhei Hikosaka. 2018.
Building Detection from Satellite Imagery using Ensemble of Size-specific Detectors.
CVPR](http://openaccess.thecvf.com/content_cvpr_2018_workshops/papers/w4/Hamaguchi_Building_Detection_From_CVPR_2018_paper.pdf)

## Environment setup

See [BUILD_ENV.md](./docs/BUILD_ENV.md).

## Download datasets

```buildoutcfg
mkdir input 
cd input
curl -k https://files.inria.fr/aerialimagelabeling/getAerial.sh | bash
```

## Execution example

See [ENDPOINTS.md](./docs/ENDPOINTS.md)