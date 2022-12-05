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


def split_dataset(train,target,validation=False):
    #70%-20%-10% split, as we're splitting 10% from the already split X_train so we're actually ending up with a 72%-20%-8% split here:
    #80 -20
    # x = img_path
    # y = 'xmin', 'ymin', 'xmax', 'ymax', 'label'

    X_train, X_test, y_train, y_test = train_test_split(train, target, train_size=0.8)

    if validation:
        X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, train_size=0.9)
        train_data = [np.squeeze(X_train, axis=1),y_train]
        validation_data = [np.squeeze(X_train, axis=1),y_valid]

        return train_data, validation_data, test_data

    train_data = [np.squeeze(X_train, axis=1),y_train]
    test_data = [np.squeeze(X_train, axis=1),y_test]

    return train_data,test_data


def train(data_loader, model, optimizer,epochs=100):
    ''' 
        sigma:  (num_samples,num_mixtures,2,2) 
        pi:     (num_samples,num_mixtures)
        mue:    (num_samples,num_mixtures,2)

        The last parameter '2' represents x and y  
    '''
    num_batches = len(data_loader)
    total_loss = 0
    model.train()
    
    for i in range (epochs):
        total_loss = 0
        for X, y in data_loader:
            #print(X.shape)
            x_variable = Variable(X,requires_grad=True).to(device)
            X = X.to(device)
            #y = y.to(device)
            pi, sigma_x,sigma_y, mu_x , mu_y = model(x_variable)
            #pi_variable, sigma_variable, mu_variable = model(x_variable)
            #print(f"sigma_variable{sigma_variable.shape}")
            loss = mdn_loss_fn( pi, sigma_x,sigma_y, mu_x , mu_y, y)
            # loss = Variable(loss, requires_grad = True)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / num_batches
        if (i+1)%5 == 0:
            print(f"Epoch {i+1} train loss: {avg_loss}")
    print(f"Epoch {i+1} train loss: {avg_loss}")
    
if __name__ == '__main__':
    # ['id', 'label', 'xmin', 'ymin', 'xmax', 'ymax', 'img_path']
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")

    IMG_FILES = glob(CFG.img_path + "/*.jpg")
    XML_FILES = glob(CFG.xml_path + "/*.xml")
    df, classes = build_df(XML_FILES)
    data = df.to_numpy()

    # input and target 
    input  = df[['img_path']]
    target = df[['xmin','ymin', 'xmax', 'ymax','label']]

    # count by groups around 1k for each class
    # group_df = df.groupby('label',sort=True).count()
    # print("group_df: ",group_df)

    # splitting data
    #train_data, validation_data, test_data = split_dataset(input,target,True)
    train_data, test_data = split_dataset(input,target,False)
    
    # Dataloaders
    train_dataloader = torch.utils.data.DataLoader(train_data, params['batch_size'],num_workers=params['num_workers'])
    #validation_dataloader = torch.utils.data.DataLoader(validation_data, params['batch_size'],num_workers=params['num_workers'])
    test_dataloader = torch.utils.data.DataLoader(test_data, params['batch_size'],num_workers=params['num_workers'])
