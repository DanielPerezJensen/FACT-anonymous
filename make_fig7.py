import numpy as np
import matplotlib.pyplot as plt
from src.util import *
from src.load_mnist import *
from load_cifar import load_cifar_classSelect
from src.GCE import GenerativeCausalExplainer
import os

# --- parameters ---
c_dim = 3
z_dim = 2
K     = 1
L     = 16
img_size = 32
class_use = np.array([7, 9])
latent_sweep_vals = np.linspace(-2,2,25)
latent_sweep_plot = [0,4,8,12,16,20,24]
classifier_path = './models/classifiers/base_cifar_79_classifier'
gce_path = './models/GCEs/base_cifar_79_gce_K1_L16_lambda005'
export_figs = True


# --- initialize ---
class_use_str = np.array2string(class_use)    
y_dim = class_use.shape[0]
newClass = range(0,y_dim)
nsweep = len(latent_sweep_vals)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# --- load test data ---
test_size = 64
X, Y, _ = load_cifar_classSelect('train', class_use, range(0, len(class_use)))
vaX, vaY, _ = load_cifar_classSelect('val', class_use, range(0, len(class_use)))
X, vaX = X / 255, vaX / 255

sample_inputs = vaX[0:test_size]
sample_labels = vaY[0:test_size]
sample_inputs_torch = torch.from_numpy(sample_inputs)
sample_inputs_torch = sample_inputs_torch.permute(0,3,1,2).float().to(device)     
ntrain = X.shape[0]


# --- load GCE ---
gcedata = torch.load(os.path.join(gce_path,'model.pt'), map_location=device)


# --- load classifier ---
from src.models.CNN_classifier import CNN
checkpoint_model = torch.load(os.path.join(classifier_path,'model.pt'), map_location=device)
classifier = CNN(y_dim, c_dim, img_size=img_size).to(device)
classifier.load_state_dict(checkpoint_model['model_state_dict_classifier'])

from src.models.CVAE import Encoder, Decoder
encoder = Encoder(K+L, c_dim, X.shape[1]**2).to(device)
decoder = Decoder(K+L, c_dim, X.shape[1]**2).to(device)
encoder.load_state_dict(gcedata['model_state_dict_encoder'])
decoder.load_state_dict(gcedata['model_state_dict_decoder'])

# --- generate latent factor sweep plot ---
sample_ind = np.concatenate((np.where(vaY == 0)[0][:1],
                             np.where(vaY == 1)[0][:1]))
cols = [[0.047,0.482,0.863],[1.000,0.761,0.039],[0.561,0.788,0.227]]
border_size = 0
nsamples = len(sample_ind)
latentsweep_vals = [-3., -2., -1., 0., 1., 2., 3.]
Xhats = np.zeros((z_dim,nsamples,len(latentsweep_vals),img_size,img_size,c_dim))
yhats = np.zeros((z_dim,nsamples,len(latentsweep_vals)))
# create plots
for isamp in range(nsamples):
    x = torch.from_numpy(np.expand_dims(vaX[sample_ind[isamp]],0))
    x_torch = x.permute(0,3,1,2).float().to(device)
    z = encoder(x_torch)[0][0].cpu().detach().numpy()
    for latent_dim in range(z_dim):
        for (ilatentsweep, latentsweep_val) in enumerate(latentsweep_vals):
            ztilde = z.copy()
            ztilde[latent_dim] += latentsweep_val
            xhat = decoder(torch.unsqueeze(torch.from_numpy(ztilde),0).to(device))
            yhat = np.argmax(classifier(xhat)[0].cpu().detach().numpy())
            img = 1.-xhat.permute(0,2,3,1).cpu().detach().numpy().squeeze()
            Xhats[latent_dim,isamp,ilatentsweep,:,:,:] = img
            yhats[latent_dim,isamp,ilatentsweep] = yhat
# format and export plots
for isamp in range(nsamples):
    fig, axs = plt.subplots(z_dim, len(latentsweep_vals))
    for latent_dim in range(z_dim):
        for (ilatentsweep, latentsweep_val) in enumerate(latentsweep_vals):
            img = Xhats[latent_dim,isamp,ilatentsweep,:,:,:].squeeze()
            axs[latent_dim,ilatentsweep].imshow(img, interpolation='nearest')
            axs[latent_dim,ilatentsweep].set_xticks([])
            axs[latent_dim,ilatentsweep].set_yticks([])
    if export_figs:
        print('Exporting sample %d/%d (%d)...' % (isamp+1, nsamples, class_use[isamp]))
        plt.savefig('./reports/figures/figure7/fig7_samp%d.svg' % (isamp), bbox_inches=0)

print('Columns - latent values in sweep: ' + str(latentsweep_vals))
print('Rows - latent dimension')