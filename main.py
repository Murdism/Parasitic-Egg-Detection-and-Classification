import os
from glob import glob
import numpy as np
import pandas as pd
from read_dataset import build_df
from utils import CFG
from sklearn.model_selection import train_test_split
import torch


# Parameters
params = {'batch_size': 64,
          'shuffle': True,
          'num_workers': 6}
max_epochs = 100


class Dataset(torch.utils.data.Dataset):
  'Characterizes a dataset for PyTorch'
  # x = img_path
  # y = 'xmin', 'ymin', 'xmax', 'ymax', 'label'


  def __init__(self, x, y):
        'Initialization'
        self.x = x
        self.y = y

  def __len__(self):
        'Denotes the total number of samples'
        return len(self.x)

  def __getitem__(self, index):
        # Load data and get label
        X = self.x[index]
        y = self.y[index]

        return X, y
def split_dataset(train,target,validation=False):
    #70%-20%-10% split, as we're splitting 10% from the already split X_train so we're actually ending up with a 72%-20%-8% split here:
    # train = train.to_numpy()
    # target = target.to_numpy()
    X_train, X_test, y_train, y_test = train_test_split(train, target, train_size=0.8)

    if validation:
        X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, train_size=0.9)
        train_data = [np.squeeze(X_train, axis=1),y_train]
        validation_data = [np.squeeze(X_train, axis=1),y_valid]

        return train_data, validation_data, test_data

    train_data = [np.squeeze(X_train, axis=1),y_train]
    test_data = [np.squeeze(X_train, axis=1),y_test]

    return train_data,test_data
if __name__ == '__main__':
    # ['id', 'label', 'xmin', 'ymin', 'xmax', 'ymax', 'img_path']

    IMG_FILES = glob(CFG.img_path + "/*.jpg")
    XML_FILES = glob(CFG.xml_path + "/*.xml")
    df, classes = build_df(XML_FILES)
    data = df.to_numpy()

    # input and target 
    input  = df[['img_path']]
    target = df[['xmin','ymin', 'xmax', 'ymax','label']]
    # print("target: ",(target.shape))

    # splitting data
    #train_data, validation_data, test_data = split_dataset(input,target,True)
    train_data, test_data = split_dataset(input,target,False)
    
    # Dataloaders
    train_dataloader = torch.utils.data.DataLoader(train_data, params['batch_size'],num_workers=params['num_workers'])
    #validation_dataloader = torch.utils.data.DataLoader(validation_data, params['batch_size'],num_workers=params['num_workers'])
    test_dataloader = torch.utils.data.DataLoader(test_data, params['batch_size'],num_workers=params['num_workers'])
