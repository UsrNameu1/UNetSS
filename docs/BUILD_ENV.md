# Build environment


## hardware prerequisite (tested environment)

- Intel Core™ i9-9900K
- Nvidia GeForce RTX 2080 Ti 

## software prerequisite

- docker: 18.09.4
- nvidia-docker: installation document is [here](https://github.com/NVIDIA/nvidia-docker)

## build docker environment

on `./docker` directory 

build image for batch

```bash
docker build -t unetss -f Dockerfile .
```

build image for notebook

```bash
docker build -t unetss/jupyter -f Dockerfile.jupyter .
```

## run container

on project root directory

run batch container

```bash
docker run --runtime=nvidia -d -it -v /$(pwd):/root -e PYTHONPATH=/root/src --name unetss unetss
```

run jupyter container

```bash
docker run --runtime=nvidia -d -it -p 8888:8888 -v /$(pwd):/tf/scripts -e PYTHONPATH=/tf/scripts --name jupyter unetss/jupyter
```
