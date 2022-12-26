###################################################################################################
# RidingData dataloader
# Victor Luder
###################################################################################################
"""
RidingData dataset
"""
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import torchvision
from torchvision import transforms
from torchvision.io import read_image

import ai8x

import os
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

"""
Custom dataset class
"""
class RidingDataDataset(Dataset):
    def __init__(self, in_data_dir, transform=None):
        labels = pd.read_csv(os.path.join(in_data_dir, "label_data_MAX78000.csv"))
        train = pd.read_csv(os.path.join(in_data_dir, "data_MAX78000.csv"))
        self.labels = labels.to_numpy()
        self.training  = train.to_numpy()
        
        self.data_dir = in_data_dir
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        l = self.labels[idx]
        label = np.expand_dims(l, axis= 0)
        t = self.training[idx, :]
        training_data = np.expand_dims(t, axis= 0)
        if self.transform:
            training_data = self.transform(torch.tensor(training_data, dtype = torch.float32))
            #label = label/(350) #expected range of labels is between 0 and 350, here it is mapped to 0 and 1
            #label = self.transform(torch.tensor(label, dtype = torch.float32))
            label = torch.tensor(label/(350), dtype = torch.float32)
        return training_data, label
        
"""
Dataloader function
"""
def ridingdata_get_datasets(data, load_train=True, load_test=True):
   
    (data_dir, args) = data
    # data_dir = data

    if load_train:
        train_transform = transforms.Compose([
             #transforms.ToTensor(),
             ai8x.normalize(args=args)
        ])
        
        train_dataset = RidingDataDataset(in_data_dir=os.path.join(data_dir, "riding_data", "train"), transform= train_transform)

    else:
        train_dataset = None

    if load_test:
    
        test_transform = transforms.Compose([
            #transforms.ToTensor(),
            ai8x.normalize(args=args)
        ])

        test_dataset = RidingDataDataset(in_data_dir=os.path.join(data_dir, "riding_data", "test"), transform=test_transform)

        # if args.truncate_testset:
        #     test_dataset.data = test_dataset.data[:1]
    else:
        test_dataset = None
    
    
    return train_dataset, test_dataset


"""
Dataset description
"""
datasets = [
    {
        'name': 'riding_data',
        'input': (1, 103),
        'output': ("power"),
        'regression': True,
        'loader': ridingdata_get_datasets,
    }
]
