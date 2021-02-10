# Regularized Neural ODEs (RNODE)
This repository contains code for reproducing the results in "MALI: A memory efficient and reverse accurate integrator for Neural ODEs".
This repository is based on the official implementation of RNODE (for the paper "How to train your Neural ODE") and ```torch_ACA``` ( for the paper "Adaptive checkpoint adjoint method for gradient estimation in Neural ODE").

## Requirements
- PyTorch 1.0+
- Install `torchdiffeq`, which provides Python CUDA ODE solvers, from [https://github.com/rtqichen/torchdiffeq](https://github.com/rtqichen/torchdiffeq)

## Data preprocessing
Follow instructions in `preprocessing/`

## Run FFJORD with MALI

### MNIST
```
cd example-scripts
sh mnist.sh
```

### CIFAR10
```
cd example-scripts
sh cifar10.sh
```

### ImageNet64
```
cd example-scripts
sh imagenet64.sh
```
