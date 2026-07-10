import torch
from torchvision import datasets, transforms
from config import N_TRAIN, SEED, K
import torch.nn.functional as F

def load_mnist_subset(n=N_TRAIN, seed=SEED):
    transform = transforms.ToTensor()
    mnist_trainset = datasets.MNIST(root='./mnist_data', train=True, download=True, transform=transform)
    mnist_testset = datasets.MNIST(root='./mnist_data', train=False, download=True, transform=transform)

    X_train_full = mnist_trainset.data.float() / 255.0
    y_train_full = mnist_trainset.targets

    X_test = mnist_testset.data.float() / 255.0
    y_test = mnist_testset.targets

    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(X_train_full.shape[0], generator=g)[:n]

    X_train = X_train_full[idx].reshape(n, -1)
    y_train = y_train_full[idx]

    X_test = X_test.reshape(X_test.shape[0], -1)
    return X_train, y_train, X_test, y_test

def onehot(y):
    return F.one_hot(y, num_classes=K).float()