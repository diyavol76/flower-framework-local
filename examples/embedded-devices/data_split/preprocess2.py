import torch
import logging
import numpy as np
from torch.utils.data import Subset
from torchvision.datasets import CIFAR10
from torchvision.transforms import Compose, Normalize, ToTensor

logging.basicConfig(format='%(levelname)s handler %(asctime)s | %(filename)s:%(lineno)d | %(message)s',
                    level=logging.DEBUG)

'''
    A very simple Data Partitioning
    implementation to be used in the Federated Learning Project.

    WARNING: This implementation is appropriate for CIFAR-10 dataset. Dataset can be changed in load_data function.

'''


class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


class DataHandler:

    def __init__(self, num_clients, split_ratio, client_id):

        """
        Initialize the DataHandler class with the dataset and client information.

        Args:
            num_clients (int): The total number of clients.
            client_id (int): The ID of the current client.
            split_ratio (float): The ratio of the training data to be used for validation.
        """
        self.client_id = client_id
        self.split_ratio = split_ratio
        self.num_clients = num_clients

    def __call__(self):
        return self.shuffle()

    def load_data(self, noise_level=0.0):
        """Load CIFAR-10 (training and test set)."""

        norm = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        transform = Compose([
            ToTensor(),
            norm,
            AddGaussianNoise(0., noise_level)])

        trainset = CIFAR10("./dataset", train=True, download=True, transform=transform)
        testset = CIFAR10("./dataset", train=False, download=True, transform=transform)

        num_examples = {"trainset": len(trainset), "testset": len(testset)}
        logging.debug(f"total length of trainset: {len(trainset)}, total length of testset: {len(testset)}")

        return trainset, testset, num_examples

    def partition(self, noise_level=0.0):

        logging.info("partition started --cid: %s - noise level: %f", self.client_id, noise_level)

        """
        Partition the dataset for the current client.

        Args:
            noise_level (float): The noise level to be added to the data.

        Returns:
            train: A subset of the partitioned train data.
            test: A subset of the partitioned test data.

        Algorithm: divides data into num_client different parts and each client takes one part
            according to its client_id.

            ex: client_id=3 takes 4th equal part of the data. (index starts from 0)

        """
        trainset, testset, num_examples = self.load_data(noise_level=noise_level)

        step_train = len(trainset) // self.num_clients
        step_test = len(testset) // self.num_clients

        train = Subset(trainset, list(range(self.client_id * step_train, (self.client_id * step_train) + step_train)))
        test = Subset(testset, list(range(self.client_id * step_test, (self.client_id * step_test) + step_test)))

        train_size = len(train)
        train_indices = list(range(train_size))
        split = int(np.floor(self.split_ratio * train_size))

        train_subset = Subset(train, train_indices[split:])
        val_subset = Subset(train, train_indices[:split])

        logging.debug(
            f"length of train subset: {len(train_subset)}, length of val subset: {len(val_subset)}, length of test subset: {len(test)}")
        return train_subset, val_subset, test

    def shuffle(self, noise_level=0.0):

        logging.info("shuffle started --cid: %s - noise level: %f", self.client_id, noise_level)

        """
        Shuffle and return a Subset for the data specific to the current client.

        Args:
            noise_level (float): The noise level to be added to the data.

        Algorithm: each client starts to take data from its client_id and takes further data with
            jumping num_clients steps.

        """
        trainset, testset, num_examples = self.load_data(noise_level=noise_level)

        mask = list(range(self.client_id, len(trainset), self.num_clients))
        train = Subset(trainset, mask)

        mask = list(range(self.client_id, len(testset), self.num_clients))
        test = Subset(testset, mask)

        train_size = len(train)
        train_indices = list(range(train_size))
        split = int(np.floor(self.split_ratio * train_size))

        train_subset = Subset(train, train_indices[split:])
        val_subset = Subset(train, train_indices[:split])

        logging.debug(
            f"length of train subset: {len(train_subset)}, length of val subset: {len(val_subset)}, length of test subset: {len(test)}")
        return train_subset, val_subset, test

    def customLabel(self, labels, noise_level=0.0):

        logging.info("custom label started --cid: %s - noise level: %f", self.client_id, noise_level)

        """
        Select data points with specific labels and return a Subset.

        Args:
            labels (list): A list of labels to be selected from the dataset.
            noise_level (float): The noise level to be added to the data.

        """
        trainset, testset, num_examples = self.load_data(noise_level=noise_level)

        train = Subset(trainset, [index for index, value in enumerate(trainset.targets) if value in labels])
        test = Subset(testset, [index for index, value in enumerate(testset.targets) if value in labels])

        train_size = len(train)
        train_indices = list(range(train_size))
        split = int(np.floor(self.split_ratio * train_size))

        train_subset = Subset(train, train_indices[split:])
        val_subset = Subset(train, train_indices[:split])

        logging.debug(
            f"length of train subset: {len(train_subset)}, length of val subset: {len(val_subset)}, length of test subset: {len(test)}")
        return train_subset, val_subset, test

    def left(self, step, noise_level=0.0):

        logging.info("%s step left shift started --cid: %s - noise level: %f", step, self.client_id, noise_level)

        """
        Return a Subset for data points using a left partitioning scheme.

        Args:
            step (int): The step size for partitioning.
            noise_level (float): The noise level to be added to the data.

        Algorithm: takes all data points with labelled idx. (idx is evaluated by 'step' step left shift)

            client_ids = 0 1 2 3
            data labels = 0 1 2 3 4 5 6 7 8 9

            if step=2, client_id=0 takes 8th, client_id=1 takes 9th, client_id=2 takes 0th, client_id=3 takes 1st  

        """
        trainset, testset, num_examples = self.load_data(noise_level=noise_level)

        idx = ((self.client_id - step + self.num_clients) % self.num_clients)
        train = Subset(trainset, [index for index, label in enumerate(trainset.targets) if label == idx])
        test = Subset(testset, [index for index, label in enumerate(testset.targets) if label == idx])

        train_size = len(train)
        train_indices = list(range(train_size))
        split = int(np.floor(self.split_ratio * train_size))

        train_subset = Subset(train, train_indices[split:])
        val_subset = Subset(train, train_indices[:split])

        logging.debug(
            f"length of train subset: {len(train_subset)}, length of val subset: {len(val_subset)}, length of test subset: {len(test)}")
        return train_subset, val_subset, test

    def right(self, step, noise_level=0.0):

        logging.info("%s step right shift started --cid: %s - noise level: %f", step, self.client_id, noise_level)

        """
        Return a Subset for data points using a right partitioning scheme.

        Args:
            step (int): The step size for partitioning.
            noise_level (float): The noise level to be added to the data.

        Algorithm: takes all data points labelled idx. (idx is evaluated by 'step' step right shift)

            client_ids = 0 1 2 3
            data labels = 0 1 2 3 4 5 6 7 8 9

            if step=2, client_id=0 takes 2nd, client_id=1 takes 3rd, client_id=2 takes 4th, client_id=3 takes 5th 

        """
        trainset, testset, num_examples = self.load_data(noise_level=noise_level)

        idx = ((self.client_id + step - self.num_clients) % self.num_clients)
        train = Subset(trainset, [index for index, label in enumerate(trainset.targets) if label == idx])
        test = Subset(testset, [index for index, label in enumerate(testset.targets) if label == idx])

        train_size = len(train)
        train_indices = list(range(train_size))
        split = int(np.floor(self.split_ratio * train_size))

        train_subset = Subset(train, train_indices[split:])
        val_subset = Subset(train, train_indices[:split])

        logging.debug(
            f"length of train subset: {len(train_subset)}, length of val subset: {len(val_subset)}, length of test subset: {len(test)}")
        return train_subset, val_subset, test

    def unbalance(self, ratio=0.2, noise_level=0.0, shuffle=False):

        var = "with shuffling" if shuffle else "with partition algorithm"
        logging.info("unbalanced distribution %s started --cid: %s - noise level: %f", var, self.client_id, noise_level)

        """
        Return a Subset for data points with unbalanced labels.

        Args:
            shuffle (bool): If True, shuffle algorithm is used to get data points else partitioning.
            ratio (float): The ratio of the data points to be cropped from the dataset (cropped label=client_id).
            noise_level (float): The noise level to be added to the data.

        Algorithm: Cropping ratio of data points with label=client_id from the dataset.

        """

        if shuffle:
            train_subset, val_subset, test = self.shuffle(noise_level=noise_level)
        else:
            train_subset, val_subset, test = self.partition(noise_level=noise_level)

        trainset = (train_subset + val_subset)

        # label_idx = [index for index, (_, label) in enumerate(trainset) if label == self.client_id]
        # other_indices = [index for index, (_, label) in enumerate(trainset) if label != self.client_id]

        # label_subset_size = int(ratio * len(label_idx))
        # selected_label_idx = torch.randperm(len(label_idx))[:label_subset_size]
        # selected_label_idx = [label_idx[i] for i in selected_label_idx]

        label_indices = [index for index, (_, label) in enumerate(trainset) if label == self.client_id]
        indices_to_keep = [index for index, (_, label) in enumerate(trainset) if label != self.client_id]

        cropped_indices = torch.randperm(len(label_indices))[:int(len(label_indices) * (1 - ratio))]
        indices_to_keep += [label_indices[i] for i in cropped_indices]

        subset_data = Subset(trainset, indices_to_keep)
        np.random.shuffle(subset_data.indices)

        train_size = len(subset_data)
        train_indices = list(range(train_size))
        split = int(np.floor(self.split_ratio * train_size))

        train_subset = Subset(subset_data, train_indices[split:])
        val_subset = Subset(subset_data, train_indices[:split])

        logging.debug(
            f"length of train subset: {len(train_subset)}, length of val subset: {len(val_subset)}, length of test subset: {len(test)}")
        return train_subset, val_subset, test
