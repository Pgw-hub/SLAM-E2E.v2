"""
Helper functions.

@author: Zhenye Na - https://github.com/Zhenye-Na
@reference: "End to End Learning for Self-Driving Cars", arXiv:1604.07316
"""

import os
import pandas as pd

from torch.utils import data
from RcCarDataset import TripletDataset

import torchvision.transforms as transforms
from torch.utils.data import DataLoader


def toDevice(datas, device):
    """
    Enable cuda.

    Args:
        datas: tensor
        device: cpu or cuda

    Returns:
        Transform `data` to `device`
    """
    imgs, angles, velocities = datas
    return imgs.float().to(device), angles.float().to(device), velocities.float().to(device)


def load_data(data_dir, train_size, csv_name, test):
    """
    Load training data and train-validation split.

    Args:
        data_dir: data root
        train_size: ratio to split to training set and validation set.

    Returns:
        trainset: training set
        valset: validation set
    """
    # reads CSV file into a single dataframe variable
    data_df = pd.read_csv(os.path.join(data_dir, csv_name),
                          names=['center', 'steering','speed'])#,'throttle', 'reverse', 'speed'])
                          
    #test하는 경우에는 testset만 return하도록 함
    if test == True:
        testset = data_df.values.tolist()
        return testset

    # Divide the data into training set and validation set
    else:
        train_len = int(train_size * data_df.shape[0])
        valid_len = data_df.shape[0] - train_len
      
        trainset, valset = data.random_split(data_df.values.tolist(), lengths=[train_len, valid_len])
        return trainset, valset


def data_loader(dataroot, trainset, valset, batch_size, shuffle, num_workers, test):
    """Self-Driving vehicles simulator dataset Loader.

    Args:
        trainset: training set
        valset: validation set
        batch_size: training set input batch size
        shuffle: whether shuffle during training process
        num_workers: number of workers in DataLoader
    Returns:
        trainloader (torch.utils.data.DataLoader): DataLoader for training set
        testloader (torch.utils.data.DataLoader): DataLoader for validation set
    """
    transformations = transforms.Compose(
        [transforms.Lambda(lambda x: (x / 127.5) - 1.0)])

    # Load training data and validation data
    
    #training_set = TripletDataset(dataroot, trainset, transformations)
    training_set = TripletDataset(dataroot, trainset, transformations, test)
    trainloader = DataLoader(training_set,
                             batch_size=batch_size,
                             shuffle=shuffle,
                             num_workers=num_workers)

    #validation_set = TripletDataset(dataroot, valset, transformations)
    validation_set = TripletDataset(dataroot, valset, transformations, test)
    valloader = DataLoader(validation_set,
                           batch_size=batch_size,
                           shuffle=shuffle,
                           num_workers=num_workers)

    return trainloader, valloader
