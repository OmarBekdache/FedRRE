import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import numpy as np
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, random_split

import augmentations
import os


def aug(image, preprocess, mixture_width, mixture_depth, aug_severity, no_jsd, all_ops):
  """Perform AugMix augmentations and compute mixture.

  Args:
    image: PIL.Image input image
    preprocess: Preprocessing function which should return a torch tensor.

  Returns:
    mixed: Augmented and mixed image.
  """
  aug_list = augmentations.augmentations
  if all_ops:
    aug_list = augmentations.augmentations_all

  ws = np.float32(np.random.dirichlet([1] * mixture_width))
  m = np.float32(np.random.beta(1, 1))

  mix = torch.zeros_like(preprocess(image))
  for i in range(mixture_width):
    image_aug = image.copy()
    depth = mixture_depth if mixture_depth > 0 else np.random.randint(
        1, 4)
    for _ in range(depth):
      op = np.random.choice(aug_list)
      image_aug = op(image_aug, aug_severity)
    # Preprocessing commutes since all coefficients are convex
    mix += ws[i] * preprocess(image_aug)

  mixed = (1 - m) * preprocess(image) + m * mix
  return mixed

class CustomDataset(Dataset):
    def __init__(self, data, labels, transform=None):
        self.data = data
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = self.data[idx]
        label = self.labels[idx]

        if self.transform:
            img = self.transform(img)

        return img, label

class AugMixDataset(torch.utils.data.Dataset):
  """Dataset wrapper to perform AugMix augmentation."""

  def __init__(self, dataset, preprocess, mixture_width, mixture_depth, aug_severity, no_jsd, all_ops):
    self.dataset = dataset
    self.preprocess = preprocess
    self.mixture_width = mixture_width
    self.mixture_depth = mixture_depth
    self.aug_severity = aug_severity
    self.no_jsd = no_jsd
    self.all_ops = all_ops

  def __getitem__(self, i):
    x, y = self.dataset[i]
    if self.no_jsd:
      return aug(x, self.preprocess), y
    else:
      im_tuple = (self.preprocess(x), aug(x, self.preprocess, self.mixture_width, self.mixture_depth, self.aug_severity, self.no_jsd, self.all_ops),
                  aug(x, self.preprocess, self.mixture_width, self.mixture_depth, self.aug_severity, self.no_jsd, self.all_ops))
      return im_tuple, y

  def __len__(self):
    return len(self.dataset)

def get_dataset(client_dataset, server_dataset, mixture_width, mixture_depth, aug_severity, no_jsd, all_ops):
    
    if client_dataset == "CIFAR10":
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        client_dataset = torchvision.datasets.CIFAR10(root='/scratch/CIFAR10', train=True, download=True, transform=transform_train)

        test_dataset = torchvision.datasets.CIFAR10(root='/scratch/CIFAR10', train=False, download=True, transform=transform_test)
        
        directory = '/scratch/cifarc/CIFAR-10-C'
        # List to store the loaded arrays
        arrays = []
        array2 = []
        testtest = np.load('/scratch/cifarc/CIFAR-10-C/labels.npy')
        # Loop through each file in the directory
        for filename in os.listdir(directory):
            if filename.endswith('.npy'):
                if filename != 'labels.npy':
                    file_path = os.path.join(directory, filename)
                    # Load the .npy file
                    data = np.load(file_path)
                    # Append the loaded array to the list
                    arrays.append(data)
                    array2.append(testtest)
        # Optionally, concatenate all arrays into a single array (if they have the same shape)
        corrupted_test_data = np.concatenate(arrays, axis=0)
        corrupted_test_labels = np.concatenate(array2, axis=0)
        corrupt_test_dataset = CustomDataset(corrupted_test_data, corrupted_test_labels, transform=transform_test)
    elif client_dataset == "CIFAR100":
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])

        client_dataset = torchvision.datasets.CIFAR100(root='/scratch/CIFAR100', train=True, download=True, transform=transform_train)

        test_dataset = torchvision.datasets.CIFAR100(root='/scratch/CIFAR100', train=False, download=True, transform=transform_test)
        
        directory = '/scratch/cifarc/CIFAR-100-C'
        # List to store the loaded arrays
        arrays = []
        array2 = []
        testtest = np.load('/scratch/cifarc/CIFAR-100-C/labels.npy')
        # Loop through each file in the directory
        for filename in os.listdir(directory):
            if filename.endswith('.npy'):
                if filename != 'labels.npy':
                    file_path = os.path.join(directory, filename)
                    # Load the .npy file
                    data = np.load(file_path)
                    # Append the loaded array to the list
                    arrays.append(data)
                    array2.append(testtest)
        # Optionally, concatenate all arrays into a single array (if they have the same shape)
        corrupted_test_data = np.concatenate(arrays, axis=0)
        corrupted_test_labels = np.concatenate(array2, axis=0)
        corrupt_test_dataset = CustomDataset(corrupted_test_data, corrupted_test_labels, transform=transform_test)
        
    if client_dataset == "AUGMIX_CIFAR10":
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip()
        ])

        preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        client_dataset = torchvision.datasets.CIFAR10(root='/scratch/CIFAR10', train=True, download=True, transform=transform_train)
        client_dataset = AugMixDataset(client_dataset, preprocess, mixture_width, mixture_depth, aug_severity, no_jsd, all_ops)

        test_dataset = torchvision.datasets.CIFAR10(root='/scratch/CIFAR10', train=False, download=True, transform=preprocess)
        
        directory = '/scratch/cifarc/CIFAR-10-C'
        # List to store the loaded arrays
        arrays = []
        array2 = []
        testtest = np.load('/scratch/cifarc/CIFAR-10-C/labels.npy')
        # Loop through each file in the directory
        for filename in os.listdir(directory):
            if filename.endswith('.npy'):
                if filename != 'labels.npy':
                    file_path = os.path.join(directory, filename)
                    # Load the .npy file
                    data = np.load(file_path)
                    # Append the loaded array to the list
                    arrays.append(data)
                    array2.append(testtest)
        # Optionally, concatenate all arrays into a single array (if they have the same shape)
        corrupted_test_data = np.concatenate(arrays, axis=0)
        corrupted_test_labels = np.concatenate(array2, axis=0)
        corrupt_test_dataset = CustomDataset(corrupted_test_data, corrupted_test_labels, transform=preprocess)
        
    if client_dataset == "AUGMIX_CIFAR100":
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip()
        ])

        preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])

        client_dataset = torchvision.datasets.CIFAR100(root='/scratch/CIFAR100', train=True, download=True, transform=transform_train)
        client_dataset = AugMixDataset(client_dataset, preprocess, mixture_width, mixture_depth, aug_severity, no_jsd, all_ops)

        test_dataset = torchvision.datasets.CIFAR100(root='/scratch/CIFAR100', train=False, download=True, transform=preprocess)
        
        directory = '/scratch/cifarc/CIFAR-100-C'
        # List to store the loaded arrays
        arrays = []
        array2 = []
        testtest = np.load('/scratch/cifarc/CIFAR-100-C/labels.npy')
        # Loop through each file in the directory
        for filename in os.listdir(directory):
            if filename.endswith('.npy'):
                if filename != 'labels.npy':
                    file_path = os.path.join(directory, filename)
                    # Load the .npy file
                    data = np.load(file_path)
                    # Append the loaded array to the list
                    arrays.append(data)
                    array2.append(testtest)
        # Optionally, concatenate all arrays into a single array (if they have the same shape)
        corrupted_test_data = np.concatenate(arrays, axis=0)
        corrupted_test_labels = np.concatenate(array2, axis=0)
        corrupt_test_dataset = CustomDataset(corrupted_test_data, corrupted_test_labels, transform=preprocess)
       
    if server_dataset == "CIFAR10":
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip()
        ])

        preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        server_dataset = torchvision.datasets.CIFAR10(root='/scratch/CIFAR10', train=True, download=True, transform=transform_train)
        server_dataset = AugMixDataset(server_dataset, preprocess, mixture_width, mixture_depth, aug_severity, no_jsd, all_ops)

        val_dataset = torchvision.datasets.CIFAR10(root='/scratch/CIFAR10', train=False, download=True)
        val_dataset = AugMixDataset(val_dataset, preprocess, mixture_width, mixture_depth, aug_severity, no_jsd, all_ops)
    elif server_dataset == "CIFAR100":
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip()
        ])

        preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])

        server_dataset = torchvision.datasets.CIFAR100(root='/scratch/CIFAR100', train=True, download=True, transform=transform_train)
        server_dataset = AugMixDataset(server_dataset, preprocess, mixture_width, mixture_depth, aug_severity, no_jsd, all_ops)

        val_dataset = torchvision.datasets.CIFAR100(root='/scratch/CIFAR100', train=False, download=True)
        val_dataset = AugMixDataset(val_dataset, preprocess, mixture_width, mixture_depth, aug_severity, no_jsd, all_ops)
        
    return client_dataset, server_dataset, test_dataset, corrupt_test_dataset, val_dataset

def split_dataset(dataset, num_clients):
    """
    Splits the dataset into `num_clients` subsets.

    Args:
        dataset (Dataset): The dataset to be split.
        num_clients (int): Number of subsets to create.

    Returns:
        list: A list of `num_clients` subsets of the original dataset.
    """
    # Calculate the approximate size of each subset
    dataset_size = len(dataset)
    client_sizes = [dataset_size // num_clients] * num_clients

    # Distribute any leftover samples across the first few clients
    for i in range(dataset_size % num_clients):
        client_sizes[i] += 1

    # Split the dataset
    client_datasets = random_split(dataset, client_sizes)
    return client_datasets

def get_loaders(client_datasets, server_dataset, test_dataset, corrupt_test_dataset, val_dataset):
    
    client_loaders = [DataLoader(client_dataset, batch_size=128, shuffle=True, num_workers=2) for client_dataset in client_datasets]
    server_loader = DataLoader(server_dataset, batch_size=128, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=100, shuffle=False, num_workers=2)
    corrupt_test_loader = DataLoader(corrupt_test_dataset, batch_size=100, shuffle=False, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=100, shuffle=False, num_workers=2)
    
    return client_loaders, server_loader, test_loader, corrupt_test_loader, val_loader
    