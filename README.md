# Skin Cancer Classification
Skin Cancer MNIST: HAM10000 (https://www.kaggle.com/kmader/skin-cancer-mnist-ham10000/home)
## Model
Resnet50

## Result
| Model                           | Prec@1       |
| ------------------------------- | ------------ |
| Resnet50                        | 0.8922155689 |
| ResNet50+Attention              | 0.8942115768 |
| ResNet50+Weighted Cross Entropy | 0.8882235529 |

## Train
```sh
  sh ./train.sh
```

## Environment
- Pytorch0.4.1
- python3.6.7
  
## Note:
**Data imbalance**

微软面试 offline test。主要是想解决数据不均衡问题(Data Augmentation + Weighted Cross Entropy)。