

# CycleGAN
Tensorflow implementation of CycleGAN.

Paper: [Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks](https://arxiv.org/pdf/1703.10593.pdf)
Author: [Jun-Yan Zhu ](https://people.eecs.berkeley.edu/~junyanz/) *et al.*


# Prerequisites
- tensorflow r1.7
- python 2.7

# Usage
```
cd cyclegan
```

## Organize Datasets
The dataset should be organized as 
datasets/doodle/
datasets/doodle/trainA
datasets/doodle/trainB
datasets/doodle/testA
datasets/doodle/testB

where trainA, testA have the hand-drawn sketch images. 
trainB, testB have the patterned-doodle images.

## Train Example
```bash
CUDA_VISIBLE_DEVICES=0 python train.py --dataset=doodle
```

## Test Example
```bash
CUDA_VISIBLE_DEVICES=0 python test.py --dataset=doodle
```
