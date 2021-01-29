#!/bin/bash
python train_classifier.py --model base --dataset mnist --class_use 3 8 --epochs 80
python train_classifier.py --model InceptionNet --dataset mnist --class_use 3 8 --epochs 80
python train_classifier.py --model ResNet --dataset mnist --class_use 3 8 --epochs 80
python train_classifier.py --model DenseNet --dataset mnist --class_use 3 8 --epochs 80

# Uncomment below to train the classifiers for the fMNIST datasets
# python train_classifier.py --model base --dataset fmnist --class_use 0 3 4 --epochs 80
# python train_classifier.py --model InceptionNet --dataset fmnist --class_use 0 3 4 --epochs 80
# python train_classifier.py --model ResNet --dataset fmnist --class_use 0 3 4 --epochs 80
# python train_classifier.py --model DenseNet --dataset fmnist --class_use 0 3 4 --epochs 80

# Note: Files found in models/classifiers/ were trained on a system with the following components:
# NVIDIA RTX 3060TI (using nvidia driver 460, and CUDA 11.2)
# AMD Ryzen 5 3600
# 32GB RAM