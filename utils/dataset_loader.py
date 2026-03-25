# Utility file to load datasets from a single interface

# Import dataset loaders
from datasets.mnist.mnist_loader import load_mnist
from datasets.fashion_mnist.fashion_mnist_loader import load_fashion_mnist
from datasets.cifar10.cifar10_loader import load_cifar10
from datasets.cifar100.cifar100_loader import load_cifar100


def load_dataset(dataset_name):
    dataset_name = dataset_name.lower()

    # MNIST dataset
    if dataset_name == "mnist":
        return load_mnist()

    # Fashion-MNIST dataset
    elif dataset_name == "fashion_mnist":
        return load_fashion_mnist()

    # CIFAR-10 dataset
    elif dataset_name == "cifar10":
        return load_cifar10()

    # CIFAR-100 dataset
    elif dataset_name == "cifar100":
        return load_cifar100()

    else:
        raise ValueError("Unsupported dataset: " + dataset_name)