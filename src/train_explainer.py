"""
    make_fig3.py
    
    Reproduces Figure 3 in O'Shaughnessy et al., 'Generative causal
    explanations of black-box classifiers,' Proc. NeurIPS 2020: global
    explanation for CNN classifier trained on MNIST 3/8 digits.
"""
import argparse
import math
import scipy.io as sio
import os
import torch
import util
import plotting
from GCE import GenerativeCausalExplainer
from load_cifar import *


def train_explainer(dataset, classes_used, K, L, lam, print_train_losses=True):
    # --- parameters ---
    # dataset
    data_classes_lst = [int(i) for i in str(classes_used)]
    dataset_name_full = dataset + '_' + str(classes_used)
    
    # classifier
    classifier_path = './pretrained_models/{}_classifier'.format(dataset_name_full)

    # GCE params
    randseed = 0
    gce_path = os.path.join('outputs', dataset_name_full + '_gce_K{}_L{}_lambda{}'.format(K, L, str(lam).replace('.', "")))
    retrain_gce = True  # train explanatory VAE from scratch
    save_gce = True  # save/overwrite pretrained explanatory VAE at gce_path
    
    # other train params
    train_steps = 3000 #8000
    Nalpha = 25
    Nbeta = 70
    batch_size = 64
    lr = 5e-4
    
    # --- initialize ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if randseed is not None:
        np.random.seed(randseed)
        torch.manual_seed(randseed)
    ylabels = range(0,len(data_classes_lst))

    # --- load data ---
    from load_mnist import load_mnist_classSelect, load_fashion_mnist_classSelect

    if dataset == 'mnist':
        fn = load_mnist_classSelect
    elif dataset == 'fmnist':
        fn = load_fashion_mnist_classSelect
    elif dataset == 'cifar':
        fn = load_cifar_classSelect
    else:
        print('dataset not correctly specified')
        
    X, Y, tridx = fn('train', data_classes_lst, ylabels)
    vaX, vaY, vaidx = fn('val', data_classes_lst, ylabels)

    if dataset == "cifar":
        X, vaX = X / 255, vaX / 255

    ntrain, nrow, ncol, c_dim = X.shape
    x_dim = nrow*ncol

    # --- load classifier ---
    from models.CNN_classifier import CNN
    classifier = CNN(len(data_classes_lst), c_dim).to(device)
    checkpoint = torch.load('%s/model.pt' % classifier_path, map_location=device)
    classifier.load_state_dict(checkpoint['model_state_dict_classifier'])
    
    
    # --- train/load GCE ---
    from models.CVAEImageNet import Decoder, Encoder
    if retrain_gce:
        encoder = Encoder(K+L, c_dim, x_dim).to(device)
        decoder = Decoder(K+L, c_dim, x_dim).to(device)
        encoder.apply(util.weights_init_normal)
        decoder.apply(util.weights_init_normal)
        gce = GenerativeCausalExplainer(classifier, decoder, encoder, device,
                                        save_output=True,
                                        save_model_params=False,
                                        save_dir=gce_path,
                                        debug_print=print_train_losses)
        traininfo = gce.train(X, K, L,
                              steps=train_steps,
                              Nalpha=Nalpha,
                              Nbeta=Nbeta,
                              lam=lam,
                              batch_size=batch_size,
                              lr=lr)
        if save_gce:
            if not os.path.exists(gce_path):
                os.makedirs(gce_path)
            torch.save(gce, os.path.join(gce_path,'model.pt'))
            sio.savemat(os.path.join(gce_path, 'training-info.mat'), {
                'data_classes_lst' : data_classes_lst, 'classifier_path' : classifier_path,
                'K' : K, 'L' : L, 'train_step' : train_steps, 'Nalpha' : Nalpha,
                'Nbeta' : Nbeta, 'lam' : lam, 'batch_size' : batch_size, 'lr' : lr,
                'randseed' : randseed, 'traininfo' : traininfo})
    else: # load pretrained model
        gce = torch.load(os.path.join(gce_path, 'model.pt'), map_location=device)
        traininfo = None
    
    # --- compute final information flow ---
    I = gce.informationFlow()
    Is = gce.informationFlow_singledim(range(0,K+L))
    print('Information flow of K=%d causal factors on classifier output:' % K)
    print(Is[:K])
    print('Information flow of L=%d noncausal factors on classifier output:' % L)
    print(Is[K:])
    
    
    # --- generate explanation and create figure ---
    nr_labels = len(data_classes_lst)
    nr_samples_fig = 8
    sample_ind = np.empty(0, dtype=int)
    
    # retrieve samples from each class
    samples_per_class = math.ceil(nr_samples_fig / nr_labels)
    for i in range(nr_labels):
        samples_per_class = math.ceil((nr_samples_fig - i * samples_per_class) / (nr_labels - i))
        sample_ind = np.int_(np.concatenate([sample_ind, np.where(vaY == i)[0][:samples_per_class]]))

    x = torch.from_numpy(vaX[sample_ind])
    zs_sweep = [-3., -2., -1., 0., 1., 2., 3.]
    Xhats, yhats = gce.explain(x, zs_sweep)
    plot_save_dir = os.path.join(gce_path, 'figs/')
    if not os.path.exists(plot_save_dir):
        os.makedirs(plot_save_dir)
    plotting.plotExplanation(1.-Xhats, yhats, save_path=plot_save_dir)

    return traininfo

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='cifar', type=str,
                        help='Name of dataset')
    parser.add_argument('--classes_used', default=35, type=str,
                        help='classes of dataset that are used')
    parser.add_argument('--K', default=1, type=int,
                        help='Number of causal factors')
    parser.add_argument('--L', default=7, type=int,
                        help='Number of noncausal factors')
    parser.add_argument('--lam', default=0.05, type=float,
                        help='Lambda parameter')
    args = parser.parse_args()
    dataset = args.dataset
    classes_used = args.classes_used
    K = args.K
    L = args.L
    lam = args.lam
    train_info = train_explainer(dataset, classes_used, K, L, lam)