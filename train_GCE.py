"""
Training function to train new GCE's with our own classifiers defined in
models.classifiers
"""

# import standard libraries
import argparse
import numpy as np
import scipy.io as sio
import os
import torch

# Import user defined libraries
from models import classifiers
from src.models.CVAE import Decoder, Encoder

import src.util as util
import src.plotting as plotting
from src.models import CNN_classifier
from src.GCE import GenerativeCausalExplainer
from src.load_mnist import *


def train_GCE(model_file, K, L, train_steps=5000,
              Nalpha=15, Nbeta=75, lam=0.05, batch_size=64,
              lr=5e-4, seed=1, retrain=False):

    save_folder_root = "models/"
    # Gather params from model name
    model_params = model_file.split("_")

    if len(model_params) != 4:
        raise InputError("model_file must be in the format: <model_name>_<data_type>_<class_use>_classifier and must be located in models/classifiers/")

    model_name = model_params[0]
    data = model_params[1]
    data_classes = np.array(list(model_params[2]), dtype=int)

    # Create path of GCE from other model
    gce_path = os.path.join(save_folder_root, "GCEs",
                            model_params[0] + "_" + model_params[1]
                            + "_" + model_params[2] + "_gce" +
                            "_K" + str(K) + "_L" + str(L) +
                            "_lambda" + str(lam).replace(".", ""))

    # Continue training if model already exists
    if not os.path.exists(gce_path + "/model.pt"):
        retrain = True

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)

    ylabels = range(0, len(data_classes))

    # Load data
    if data.lower() == "mnist":
        X, Y, tridx = load_mnist_classSelect('train', data_classes, ylabels)
        vaX, vaY, vaidx = load_mnist_classSelect('val', data_classes, ylabels)
    elif data.lower() == "fmnist":
        X, Y, tridx = load_fashion_mnist_classSelect('train', data_classes, ylabels)
        vaX, vaY, vaidx = load_fashion_mnist_classSelect('val', data_classes, ylabels)
    elif data.lower() == 'cifar':
        from load_cifar import load_cifar_classSelect
        X, Y, _ = load_cifar_classSelect('train', data_classes, ylabels)
        vaX, vaY, _ = load_cifar_classSelect('val', data_classes, ylabels)
        X, vaX = X / 255, vaX / 255

    ntrain, nrow, ncol, c_dim = X.shape
    x_dim = nrow * ncol
    y_dim = len(data_classes)

    # Import stated model
    if model_name.lower() == "inceptionnet":
        classifier = classifiers.InceptionNetDerivative(num_classes=y_dim).to(device)
    elif model_name.lower() == "resnet":
        classifier = classifiers.ResNetDerivative(num_classes=y_dim).to(device)
    elif model_name.lower() == "densenet":
        classifier = classifiers.DenseNetDerivative(num_classes=y_dim).to(device)
    elif model_name.lower() == "base":
        classifier = CNN_classifier.CNN(y_dim, c_dim, img_size=nrow).to(device)

    # Load previously trained classifier
    checkpoint = torch.load('%s/model.pt' % (save_folder_root + "/classifiers/" + model_file), map_location=device)
    classifier.load_state_dict(checkpoint['model_state_dict_classifier'])

    # Train a new model
    if retrain:
        # Declare GCE and it's needed variables
        encoder = Encoder(K+L, c_dim, x_dim).to(device)
        decoder = Decoder(K+L, c_dim, x_dim).to(device)
        encoder.apply(util.weights_init_normal)
        decoder.apply(util.weights_init_normal)
        gce = GenerativeCausalExplainer(classifier, decoder, encoder, device,
                                        save_output=True, save_dir=gce_path + "/")

        traininfo = gce.train(X, K, L,
                              steps=train_steps,
                              Nalpha=Nalpha,
                              Nbeta=Nbeta,
                              lam=lam,
                              batch_size=batch_size,
                              lr=lr)
        torch.save({
            "model_state_dict_classifier": gce.classifier.state_dict(),
            "model_state_dict_encoder": gce.encoder.state_dict(),
            "model_state_dict_decoder": gce.decoder.state_dict(),
            "step": train_steps
        }, os.path.join(gce_path, 'model.pt'))

    # Continue training a partly trained model
    else:
        encoder = Encoder(K+L, c_dim, x_dim).to(device)
        decoder = Decoder(K+L, c_dim, x_dim).to(device)
        # Load GCE from stored model
        gce = GenerativeCausalExplainer(classifier, decoder, encoder, device,
                                        save_output=True, save_dir=gce_path + "/")
        checkpoint = torch.load(os.path.join(gce_path, 'model.pt'), map_location=device)

        gce.classifier.load_state_dict(checkpoint["model_state_dict_classifier"])
        gce.encoder.load_state_dict(checkpoint["model_state_dict_encoder"])
        gce.decoder.load_state_dict(checkpoint["model_state_dict_decoder"])

        if checkpoint["step"] < train_steps:
            print(f"Continuing training previous model from step: {checkpoint['step']}")
            traininfo = gce.train(X, K, L,
                                  steps=train_steps - checkpoint["step"],
                                  Nalpha=Nalpha,
                                  Nbeta=Nbeta,
                                  lam=lam,
                                  batch_size=batch_size,
                                  lr=lr)

            os.makedirs(gce_path, exist_ok=True)

            torch.save({
                "model_state_dict_classifier": gce.classifier.state_dict(),
                "model_state_dict_encoder": gce.encoder.state_dict(),
                "model_state_dict_decoder": gce.decoder.state_dict(),
                "step": train_steps + checkpoint["step"]
            }, os.path.join(gce_path, 'model.pt'))

        else:
            raise ValueError(f"Not continuing training previous model since {checkpoint['step']} >= {train_steps}")

    return traininfo


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_file", type=str, default="base_cifar_35_classifier",
                        help="Specification of path to model to be explained by GCE.")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Specification of batch size to be used.")
    parser.add_argument("--train_steps", type=int, default=3000,
                        help="Specification of training steps for GCE.")
    parser.add_argument("--lr", type=float, default=5e-4,
                        help="Specification of learning rate")
    parser.add_argument("--K", type=int, default=1,
                        help="Specification of number of causal Factors")
    parser.add_argument("--L", type=int, default=16,
                        help="Specification of number of non-causal Factors")
    parser.add_argument("--lam", type=float, default=0.05,
                        help="Specification of lambda parameter")
    parser.add_argument("--Nalpha", type=int, default=15,
                        help="Specification of number of samples to estimate alpha")
    parser.add_argument("--Nbeta", type=int, default=75,
                        help="Specification of number of samples to estimate beta")
    parser.add_argument("--seed", type=int, default=1,
                        help="Specification of random seed of this run")

    args = parser.parse_args()

    # Training Parameters
    K = args.K
    L = args.L
    train_steps = args.train_steps
    Nalpha = args.Nalpha
    Nbeta = args.Nbeta
    lam = args.lam
    batch_size = args.batch_size
    lr = args.lr

    seed = args.seed

    train_GCE(args.model_file, K, L, train_steps, Nalpha, Nbeta, lam, batch_size, lr, seed)
