import numpy as np
import torchvision
import torchvision.transforms as trans
def load_cifar_classSelect(data_type, class_use, newClass, gray=False):
    if data_type == 'train':
        dataset = torchvision.datasets.CIFAR10(root='./datasets/cifar', train=True, download=True)
        X = dataset.data[:int(len(dataset.targets)*0.8)]
        Y = np.array(dataset.targets[:int(len(dataset.targets)*0.8)])

    elif data_type == 'val':
        dataset = torchvision.datasets.CIFAR10(root='./datasets/cifar', train=True, download=True)
        X = dataset.data[int(len(dataset.targets) * 0.8):]
        Y = np.array(dataset.targets[int(len(dataset.targets) * 0.8):])

    else:
        dataset = torchvision.datasets.CIFAR10(root='./datasets/cifar', train=False, download=True)
        X = dataset.data
        Y = np.array(dataset.targets)

    idx = np.concatenate([np.where(Y == _class)[0] for _class in class_use])

    np.random.shuffle(idx)

    trainDataUse, Y = X[idx], np.array(Y)[idx]

    trainTargetsUse = Y[:]

    count_y = 0
    for k in class_use:
        class_idx = np.where(Y == k)[0]
        trainTargetsUse[class_idx] = newClass[count_y]
        count_y += 1

    trainDataUse = np.expand_dims(np.dot(trainDataUse[..., :3], [0.2989, 0.5870, 0.1140]), 3) if gray else trainDataUse
    return trainDataUse, trainTargetsUse, 0