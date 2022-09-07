# EPN-NetVLAD
This repository contains the code (in PyTorch) for **SE(3)-Equivariant Point Cloud-based Place Recognition**.

## Abstract
This paper reports on a new 3D point cloud-based place recognition framework that uses SE(3)-equivariant networks to learn SE(3)-invariant global descriptors. We discover that, unlike existing methods, learned SE(3)-invariant global descriptors are more robust to matching inaccuracy and failure in severe rotation and translation configurations. Mobile robots undergo arbitrary rotational and translational movements. The SE(3)-invariant property ensures the learned descriptors are robust to the rotation and translation changes in the robot pose and can represent the intrinsic geometric information of the scene. Furthermore, we have discovered that the attention module aids in the enhancement of performance while allowing significant downsampling. We evaluate the performance of the proposed framework on real-world data sets. The experimental results show that the proposed framework outperforms state-of-the-art baselines in various metrics, leading to a reliable point cloud-based place recognition network.
![](media/se3_equivariant_place_recognition.png)

## Set Up
See [docker](docker) folder for how to use docker image and build docker container.
In addition to docker, The module and additional dependencies can be installed with
```
cd vgtk
python setup.py install
```
Note: It might require root access to install the module.

## Experiments

### Datasets
This repository is tested with Oxford Robocar benchmark created by [PointNetVLAD](https://github.com/mikacuy/pointnetvlad), and can be downloaded [here](https://drive.google.com/drive/folders/1Wn1Lvvk0oAkwOUwR0R6apbrekdXAUg7D). 

We create training tuples following [PointNetVLAD](https://github.com/mikacuy/pointnetvlad)'s procedure to generate training and testing tuples in pickle files. The picke files we used for training and testing will be will be publicly available after receiving the final decision. Please see [PointNetVLAD's generating_queries folder](https://github.com/mikacuy/pointnetvlad/tree/master/generating_queries) for detail implementation.

### Pretrained Model
A pretrained weight for `EPN-NetVLAD` model with or without attentive downsampling will be will be publicly available after receiving the final decision.

### Training
After changing the cofigurations for training settings in [config.py](config.py) file, the following command can be used to train the model:

```
python run_oxford.py
```

### Evaluation
After changing the cofigurations for evaluation settings in [config.py](config.py) file, the following command can be used for the evaluation:

```
python evaluate_place_recognition.py
```

### Results
Results show the precision-recall curve, f1-recall curve, and average recall at top N curve.
![](media/precision_recall_curve.png)
![](media/f1_recall_curve.png)
![](media/average_recall_curve.png)

## Reference Code
- [EPN-PointCloud](https://github.com/nintendops/EPN_PointCloud): Equivariant Point Network (EPN). We modified it to be our SE(3)-invariant point cloud local feature extractor in our framework.
- [PointNetVLAD in Tensorflow](https://github.com/mikacuy/pointnetvlad) and [PointNetVlad-Pytorch](https://github.com/cattaneod/PointNetVlad-Pytorch). We utilized the benchmark_dataset, code for training with quadruplet loss, and place recognition evaluation.