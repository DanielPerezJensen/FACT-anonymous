# FACT
Repository for our group project for the FACT course taken as part of the Master of AI at the UVA.

Reproducibility of"[Generative causal explanations of black-box classifiers](https://arxiv.org/abs/2006.13913)" by Matt O'Shaughnessy, Greg Canal, Marissa Connor, Mark Davenport, and Chris Rozell (Proc. NeurIPS 2020).

# Code Structure
```
.
├── AUTHORS.md
├── README.md
├── algorithm1.py <- algorithm 1 as defined in the original paper
├── load_cifar.py <- code for loading and using the cifar10 dataset
├── train_GCE.py <- training code for the general causal explainers
├── train_classifier.py <- training code for classifiers
├── train_GCEs.sh
├── train_classifiers.sh
├── make_fig7.py
├── generate_explanations.py <- code for generating sweeps across latent space
├── models  <-
	├── GCEs <- compiled general causal explainers
	├── classifiers <- compiled classifiers
	├── classifiers.py <- python modules defining classifiers
├── datasets <- batch files with used datasets
    ├── mnist
    ├── fmnist
    ├── cifar10 <- will be created after training a model using cifar10
├── reports <- generated project artefacts eg. visualisations or tables
    ├── figures <- all generated figures can be found here
    ├── params <- all parameter data can be found here
    ├── results <- figures that are presented in our report
    ├── results.ipynb <- notebook with all figures as presented in our report
├── src <- files gathered from repo of original paper
    ├── ...
    ├── ... 
|--- .gitignore <- file with libraries and library versions for recreating the environment
|--- requirements.txt <- file with libraries and library versions for recreating the environment
```

# Dependencies
All dependencies can be installed using the provided `requirements.txt` file. These dependencies can be installed using `pip` and can be installed like so: `pip install -r requirements.txt`.

# Training code
## Classifiers
To train any of the 4 provided classifiers you can use the `train_classifier.py` script. This scripts has multiple optional arguments, all these arguments can be gathered by running: `python train_classifier.py --help`.

For example to train a classifier on MNIST for the classes 3 and 8 using InceptionNet you could call:

`python train_classifier.py --model inceptionnet --dataset mnist -- class_use 3 8`

Options not listed will use the default parameters listed in the bottom of the `train_classifier.py` file.

To train all classifiers as they were used to create the figures found in our reproducibility report please run: `bash train_classifiers.sh`. This will launch a bash script that runs the training script for all the classifiers we used ourselves.

To train your own classifier you first need to create a module (using Pytorch) for it in `models/classifiers.py`. The forward function should return the same as the forward functions of our own classifiers. To call this classifier you could add an option in `train_classifiers.py` from line 84:

```python
# Import stated model
if model_name.lower() == "inceptionnet":
    classifier = classifiers.InceptionNetDerivative(num_classes=y_dim).to(device)
...
...
...

elif model_name.lower() == "your-own-model":
    classifier = classifiers.YourOwnModel(num_classes=y_dim).to(device)
...
```

You can then run:

```bash 
python train_classifier.py --model your-own-model ...
```

The classifiers will be stored in `models/classifiers/`.

## Explainers

To train an explainer for a classifier you can use `train_GCE.py`. This script also has optional arguments, which can be listed by running: `python train_GCE.py --help`. The default parameters can likewise be found at the bottom of the file.

To for example train the inceptionnet classifier we trained above with 1 alpha factor and 7 beta factors, with a lambda of 0.05 you could call:

```bash
python train_GCE --model_file inceptionnet_mnist_38_classifier --K 1 --L 7 --lam 0.05
```

To train all explainers we used to create our figures, you can run: `bash train_GCEs.sh`. This will launch a bash script that retrains all the models with the hyperparameters as they are found in our report.

Again this script can be used to work with your own classifier you just need to specify the correct `--model_file`. I.e. `python train_GCE.py --model_file your-own-model-classifier ...`

# Figures
## Sweeps
To generate the sweeps as are shown in for example Figure 3 you can use `generate_explanations.py`. It has one optional argument where you can specify what explainer to use to create the sweep. The default parameter is listed at the bottom of the file. To for example re-create the figure using the Inception-Net classifier we trained earlier you can run:

```bash
python generate_explanations.py --model_file inceptionnet_mnist_38_gce_K1_L7_lambda005
```

This can be used to create explanations for any explainer, as long as their `model.pt` file is located in `models/GCEs`. To for example us your own trained classifier and explainer you can run:

```bash
python generate_explanations.py --model_file your-own-model-gce_.....
```

### Figure 2
To re-create the sweeps as shown in Figure 2 you can run:

```bash 
python generate_explanations.py --model_file inceptionnet_mnist_38_gce_K1_L7_lambda005
```

This by default uses our provided trained models. If you first want to retrain the classifier and GCE run (assuming you didn't run the provided bash scripts already):

```bash 
python train_classifier.py --model inceptionnet --dataset mnist --class_use 3 8 --epochs 80
python train_GCE.py --model_file inceptionnet_mnist_38_classifier --train_steps 8000 --K 1 --L 7 --lam 0.05 --Nalpha 15 --Nbeta 75
```

Followed by:
```bash
python generate_explanations.py --model_file inceptionnet_mnist_38_gce_K1_L7_lambda005
```

### Figure 3
To re-create the sweeps as shown in Figure 3 you can run:
```bash
python generate_explanations.py --model_file resnet_mnist_38_gce_K1_L7_lambda005
```

This by default uses our provided trained models. If you first want to retrain the classifier and GCE run (assuming you didn't run the provided bash scripts already):

```bash
python train_classifier.py --model resnet --dataset mnist --class_use 3 8 --epochs 80
python train_GCE.py --model_file resnet_mnist_38_classifier --train_steps 8000 --K 1 --L 7 --lam 0.05 --Nalpha 15 --Nbeta 75
```

Followed by:
```bash
python generate_explanations.py --model_file resnet_mnist_38_gce_K1_L7_lambda005
```

### Figure 4
To re-create the sweeps as shown in Figure 4 you can run:
```bash
python generate_explanations.py --model_file base_fmnist_034_gce_K2_L4_lambda004
```

This by default uses our provided trained models. If you first want to retrain the classifier and GCE run (assuming you didn't run the provided bash scripts already):

```bash
python train_classifier.py --model base --dataset fmnist --class_use 0 3 4 --epochs 80
python train_GCE.py --model_file base_fmnist_034_classifier --train_steps 3000 --K 2 --L 4 --lam 0.04 --Nalpha 15 --Nbeta 75
```

Followed by:
```bash
python generate_explanations.py --model_file base_fmnist_034_gce_K2_L4_lambda004
```

### Figure 5
To re-create the sweeps as shown in Figure 5 you can run:
```bash
python generate_explanations.py --model_file base_fmnist_034_gce_K2_L4_lambda007
```

This by default uses our provided trained models. If you first want to retrain the classifier and GCE run (assuming you didn't run the provided bash scripts already):

```bash
python train_classifier.py --model base --dataset fmnist --class_use 0 3 4 --epochs 80
python train_GCE.py --model_file base_fmnist_38_classifier --train_steps 3000 --K 2 --L 4 --lam 0.07 --Nalpha 15 --Nbeta 75
```

Followed by:
```bash
python generate_explanations.py --model_file base_fmnist_034_gce_K2_L4_lambda007
```


### Figure 6
To re-create the sweeps as shown in Figure 6 you can run:
```bash
python generate_explanations.py --model_file base_cifar_79_gce_K1_L16_lambda005
```

This by default uses our provided trained models. If you first want to retrain the classifier and GCE run (assuming you didn't run the provided bash scripts already):

```bash
python train_classifier.py --model base --dataset cifar --class_use 7 9 --epochs 150
python train_GCE.py --model_file base_cifar_79_classifier --train_steps 4000 --K 1 --L 16 --lam 0.05 --Nalpha 20 --Nbeta 70
```

Followed by:
```bash
python generate_explanations.py --model_file base_cifar_79_gce_K1_L16_lambda005
```

### Figure 7
To re-create the alpha sweep of Figure 7 you can run:
```bash
python make_fig7.py
```
