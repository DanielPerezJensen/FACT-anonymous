#!/usr/bin/env python

"""Python functions that train the classifiers descrbied in
model/classifiers.py using either MNIST or fMNIST"""

# Standard libraries
import argparse
import numpy as np
import os
import time

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import StepLR


import datetime

# User-made scripts
from models import classifiers

from src.models import CNN_classifier
from src.load_mnist import *
from src.mnist_reader import *


def main():

    print(args)

    # (hyper)parameters
    model_name = args.model
    dataset = args.dataset
    class_use = np.array(args.class_use)
    batch_size = args.batch_size
    epochs = args.epochs
    lr = args.lr
    momentum = args.momentum

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    class_use_str = "".join(map(str, class_use))
    y_dim = class_use.shape[0]
    newClass = range(0, y_dim)
    test_size = 100

    save_folder_root = './models/classifiers/'
    save_folder = os.path.join(save_folder_root,
                               f"{model_name.lower()}_{dataset}_{class_use_str}_classifier")
    summary_writer = SummaryWriter(save_folder + "/runs")

    os.makedirs(save_folder, exist_ok=True)
    os.makedirs(save_folder + "/runs", exist_ok=True)

    # Dataset preparation
    if dataset == 'mnist':
        trX, trY, tridx = load_mnist_classSelect('train', class_use, newClass)
        vaX, vaY, vaidx = load_mnist_classSelect('val', class_use, newClass)
        teX, teY, teidx = load_mnist_classSelect('test', class_use, newClass)
    elif dataset == 'fmnist':
        trX, trY, tridx = load_fashion_mnist_classSelect('train', class_use, newClass)
        vaX, vaY, vaidx = load_fashion_mnist_classSelect('val', class_use, newClass)
        teX, teY, teidx = load_fashion_mnist_classSelect('test', class_use, newClass)
    elif dataset == 'cifar':
        from load_cifar import load_cifar_classSelect
        trX, trY, _ = load_cifar_classSelect('train', class_use, newClass)
        vaX, vaY, _ = load_cifar_classSelect('val', class_use, newClass)
        trX, vaX = trX, vaX
    else:
        print('dataset must be ''cifar'' ''mnist'' or ''fmnist''!')

    c_dim = trX.shape[-1]
    img_size = trX.shape[1]

    # Number of batches per epoch
    batch_idxs = len(trX) // batch_size
    batch_idxs_val = len(vaX) // test_size

    # Training
    ce_loss = nn.CrossEntropyLoss()

    # Import stated model
    if model_name.lower() == "inceptionnet":
        classifier = classifiers.InceptionNetDerivative(num_classes=y_dim).to(device)
    elif model_name.lower() == "resnet":
        classifier = classifiers.ResNetDerivative(num_classes=y_dim).to(device)
    elif model_name.lower() == "densenet":
        classifier = classifiers.DenseNetDerivative(num_classes=y_dim).to(device)
    elif model_name.lower() == "base":
        classifier = CNN_classifier.CNN(y_dim, c_dim, img_size).to(device)
    else:
        raise ValueError("Invalid model_name, options=['InceptionNet', 'ResNet', 'DenseNet']")

    optimizer = torch.optim.SGD(classifier.parameters(), lr=lr, momentum=momentum)
    scheduler = StepLR(optimizer, step_size=1, gamma=0.7)

    best_loss = np.inf

    for epoch in range(epochs):

        total_train_loss = 0.0
        total_train_acc = 0

        total_test_loss = 0.0
        total_test_acc = 0

        for idx in range(batch_idxs):
            classifier.train()
            # Gather input and labels for current batch
            labels = torch.from_numpy(trY[idx * batch_size:(idx + 1) * batch_size]).long().to(device)
            inp = trX[idx * batch_size:(idx + 1) * batch_size]
            inp = torch.from_numpy(inp)
            inp = inp.permute(0, 3, 1, 2).float()
            inp = inp.to(device)

            # Calculate loss and acc, then step with optimizer
            optimizer.zero_grad()
            probs, out = classifier(inp)
            loss = ce_loss(out, labels)
            loss.backward()

            acc = (probs.argmax(dim=-1) == labels).float().mean()
            optimizer.step()

            # Add loss and accuracy to acumulators
            total_train_loss += loss.item()
            total_train_acc += acc

            print("[{}] Train Epoch {:02d}/{:02d}, Batch Size = {}, \
                  train loss = {:.3f}, train acc = {:.2f}".format(
                    datetime.datetime.now().strftime("%Y-%m-%d %H:%M"), epoch + 1,
                    epochs, batch_size, loss.item(), acc))

        total_train_loss /= batch_idxs
        total_train_acc /= batch_idxs

        summary_writer.add_scalar("Loss/train", total_train_loss, epoch)
        summary_writer.add_scalar("Acc/train", total_train_acc, epoch)

        for idx in range(batch_idxs_val):
            classifier.eval()
            # Gather validation input and labels
            val_labels = torch.from_numpy(vaY[idx * test_size:(idx + 1) * test_size]).long().to(device)
            val_inp = vaX[idx * test_size:(idx + 1) * test_size]
            val_inp = torch.from_numpy(val_inp)
            val_inp = val_inp.permute(0, 3, 1, 2).float()
            val_inp = val_inp.to(device)

            probs, out = classifier(val_inp)
            total_test_loss += ce_loss(out, val_labels)
            total_test_acc += (probs.argmax(dim=-1) == val_labels).float().mean()

        total_test_loss /= batch_idxs_val
        total_test_acc /= batch_idxs_val

        summary_writer.add_scalar("Loss/val", total_test_loss, epoch)
        summary_writer.add_scalar("Acc/val", total_test_acc, epoch)

        if total_test_loss < best_loss:
            print(f"New best model, saving to: {save_folder}")

            torch.save({
                    "epoch": epoch,
                    "model_state_dict_classifier": classifier.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                }, os.path.join(save_folder, "model.pt"))

        print("[{}] Test Epoch {:02d}/{:02d}, \
              test loss = {:.3f}, test acc = {:.2f}".format(
                datetime.datetime.now().strftime("%Y-%m-%d %H:%M"), epoch + 1,
                epochs, total_test_loss, total_test_acc))

        scheduler.step()

    summary_writer.flush()
    summary_writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", type=str, default="base",
                        help="Specification of model to be trained.")
    parser.add_argument("--dataset", type=str, default="cifar",
                        help="Specification of dataset to be used.")
    parser.add_argument("--class_use", type=int, default=[3, 5], nargs="+",
                        help="Specification of what classes to use" +
                             "To specify multiple, use \" \" to" +
                             "separate them. Example \"3 8\".")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Specification of batch size to be used.")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Specification of training epochs.")
    parser.add_argument("--lr", type=float, default=0.0005,
                        help="Specification of learning rate for SGD")
    parser.add_argument("--momentum", type=float, default=0.9,
                        help="Specification of Momentum for SGD")
    parser.add_argument("--seed", type=int, default=1,
                        help="Specification of random seed of this run")

    args = parser.parse_args()

    main()
