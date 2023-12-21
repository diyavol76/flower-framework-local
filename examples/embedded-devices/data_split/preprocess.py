import logging
import numpy as np
from torch.utils.data import Subset
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms

from torchvision.transforms import Compose, Normalize, ToTensor

logging.basicConfig(format='%(levelname)s handler %(asctime)s | %(filename)s:%(lineno)d | %(message)s', level=logging.DEBUG)

'''
    A very simple Data Partitioning
    implementation to be used in the Federated Learning Project.
    
    WARNING: This implementation is appropriate for CIFAR-10 dataset. Dataset can be changed in load_data function.

'''

class DataHandler:
    
    def __init__(self, num_clients, split_ratio, client_id):
        
        """
        Initialize the DataHandler class with the dataset and client information.
        
        Args:
            num_clients (int): The total number of clients.
            client_id (int): The ID of the current client.
        """
        self.client_id=client_id
        self.split_ratio=split_ratio
        self.num_clients=num_clients
       
    def __call__(self): return self.shuffle()
        
    def load_data(self):
        """Load CIFAR-10 (training and test set)."""
        transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),

            ]
        )
        norm = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        trf = Compose([ToTensor(), norm])
        trainset = CIFAR10("./dataset", train=True, download=True, transform=trf)
        testset = CIFAR10("./dataset", train=False, download=True, transform=trf)

        num_examples = {"trainset": len(trainset), "testset": len(testset)}
        return trainset, testset, num_examples
        
    def partition(self):
        
        logging.debug("partition started --cid: %s", self.client_id)
        
        """
        Partition the dataset for the current client.

        Returns:
            train: A subset of the partitioned train data.
            test: A subset of the partitioned test data.
            
        Algorithm: divides data into num_client different parts and each client takes one part
            according to its client_id.
            
            ex: client_id=3 takes 4th equal part of the data. (index starts from 0)
            
        """
        trainset, testset, num_examples = self.load_data()
        
        step_train = len(trainset) // self.num_clients
        step_test = len(testset) // self.num_clients
        
        train = Subset(trainset, list(range(self.client_id * step_train, (self.client_id * step_train) + step_train))) 
        test =  Subset(testset, list(range(self.client_id * step_test, (self.client_id * step_test) + step_test)))

        train_size = len(train)
        train_indices = list(range(train_size))
        split = int(np.floor(self.split_ratio * train_size))

        train_subset = Subset(train, train_indices[:split])
        val_subset = Subset(train, train_indices[split:])
        
        return train_subset, val_subset, test

    def shuffle(self):  
        
        logging.debug("shuffle started --cid: %s", self.client_id)
        
        """
        Shuffle and return a DataLoader for the data specific to the current client.
        
        Algorithm: each client starts to take data from its client_id and takes further data with
            jumping num_clients steps.
            
        """
        trainset, testset, num_examples = self.load_data()
        
        mask = list(range(self.client_id, len(trainset), self.num_clients))
        train = Subset(trainset, mask)
        
        mask = list(range(self.client_id, len(testset), self.num_clients))
        test = Subset(testset, mask)

        train_size = len(train)
        train_indices = list(range(train_size))
        split = int(np.floor(self.split_ratio * train_size))

        train_subset = Subset(train, train_indices[:split])
        val_subset = Subset(train, train_indices[split:])
        
        return train_subset, val_subset, test
        
    def customLabel(self, labels): 
        
        logging.debug("custom label started --cid: %s", self.client_id)    
        
        """
        Select data points with specific labels and create data loaders.
        
        Args:
            labels (list): A list of labels to be selected from the dataset.
        
        """       
        trainset, testset, num_examples = self.load_data()
        
        train = Subset(trainset, [index for index, value in enumerate(trainset.targets) if value in labels])
        test = Subset(testset, [index for index, value in enumerate(testset.targets) if value in labels])

        train_size = len(train)
        train_indices = list(range(train_size))
        split = int(np.floor(self.split_ratio * train_size))

        train_subset = Subset(train, train_indices[:split])
        val_subset = Subset(train, train_indices[split:])
        
        return train_subset, val_subset, test    
            
    def left(self, step):
        
        logging.debug("%s step left shift started --cid: %s", step, self.client_id)
        
        """
        Select and return a DataLoader for data points using a left partitioning scheme.

        Args:
            step (int): The step size for partitioning.
            
        Algorithm: takes all data points with labelled idx. (idx is evaluated by 'step' step left shift)
        
            client_ids = 0 1 2 3
            data labels = 0 1 2 3 4 5 6 7 8 9
            
            if step=2, client_id=0 takes 8th, client_id=1 takes 9th, client_id=2 takes 0th, client_id=3 takes 1st  
                
        """
        trainset, testset, num_examples = self.load_data()
    
        idx = ((self.client_id - step + self.num_clients) % self.num_clients)
        train = Subset(trainset, [index for index, label in enumerate(trainset.targets) if label == idx])
        test = Subset(testset, [index for index, label in enumerate(testset.targets) if label == idx])

        train_size = len(train)
        train_indices = list(range(train_size))
        split = int(np.floor(self.split_ratio * train_size))

        train_subset = Subset(train, train_indices[:split])
        val_subset = Subset(train, train_indices[split:])
        
        return train_subset, val_subset, test
    
    def right(self, step):
        
        logging.debug("%s step right shift started --cid: %s", step, self.client_id)
        
        """
        Select and return a DataLoader for data points using a right partitioning scheme.

        Args:
            step (int): The step size for partitioning.
            
        Algorithm: takes all data points labelled idx. (idx is evaluated by 'step' step right shift)
        
            client_ids = 0 1 2 3
            data labels = 0 1 2 3 4 5 6 7 8 9
            
            if step=2, client_id=0 takes 2nd, client_id=1 takes 3rd, client_id=2 takes 4th, client_id=3 takes 5th 

        """
        trainset, testset, num_examples = self.load_data()
    
        idx = ((self.client_id + step - self.num_clients) % self.num_clients)
        train = Subset(trainset, [index for index, label in enumerate(trainset.targets) if label == idx])
        test = Subset(testset, [index for index, label in enumerate(testset.targets) if label == idx])
        
        train_size = len(train)
        train_indices = list(range(train_size))
        split = int(np.floor(self.split_ratio * train_size))

        train_subset = Subset(train, train_indices[:split])
        val_subset = Subset(train, train_indices[split:])
        
        return train_subset, val_subset, test
