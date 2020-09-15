# semi-self-supervised

This is an official PyTorch implementation of [Wafer BIN Map Defect Image Classification with Semi-Self-Supervised Learning](https://arxiv.org/abs/test).

This code is only available in Semi-Self Supervised Learning.
Now only experiments on WM-811K is available.

## Requirements
- Python 3.6+
- PyTorch 1.4.0
- torchvision 0.5
- tensorboard
- tqdm
- numpy

## Usage

### Train
Train the model by 225, 450, 900 labeled data of WM-811K dataset:

```
python train.py --dataset wm811k --num-labeled 900 --arch wideresnet --batch-size 64 --lr 0.03 --out wm-811k@900 (TODO)
```

Train the model by 900 labeled data of Wm-811K dataset by using DistributedDataParallel:
```
python -m torch.distributed.launch --nproc_per_node 4 ./train.py --dataset wm811k --num-labeled 900 --arch wideresnet --batch-size 16 --lr 0.03 --out wm811k@900
```

### Monitoring training progress
```
tensorboard --logdir=<your out_dir>
```

## Results (Accuracy)

### WM-811K
| #Labels | 225 | 450 | 900 |
|:---|:---:|:---:|:---:|
|Supervised | 23.51 ± 2.36 | 19.39 ± 4.66 | 10.26 ± 2.98 |
|Pseudo-Label | 7.23 ± 1.35 | 5.98 ± 0.21 | 4.94 ± 0.07 |
|PI-Model | 7.42 ± 0.99 | 6.04 ± 0.41 | 5.08 ± 0.02 |
|Mean-Teacher | 7.64 ± 0.86 | 7.49 ± 1.19 | 5.13 ± 0.18 |
|VAT | 8.43 ± 1.1 | 5.97 ± 0.3 | 8.3 ± 0.53 |
|VAT + EntMin. | 8.6 ± 1.98 | 6.1 ± 0.31 | 5.98 ± 0.73 |
|FixMatch (RA) | 13.14 ± 1.87 | 8.99 ± 1 | 5.49 ± 0.72 |
|Paper (our) | 6.03 ± 0.14 | 5.35 ± 0.17 | 4.66 ± 0.13 |

### CIFAR10
| #Labels | 40 | 250 | 4000 |
|:---|:---:|:---:|:---:|
|Supervised | - | - | - |
|Pseudo-Label | - | - | - |
|PI-Model | - | - | - |
|Mean-Teacher | - | - | - |
|VAT | - | - | - |
|VAT + EntMin. | - | - | - |
|FixMatch (RA) | 86.19 ± 3.37 | 94.93 ± 0.65 | 95.74 ± 0.05 |
|Paper (our) |  |  |  |


\* Results of this code were evaluated on 1 run.


```
