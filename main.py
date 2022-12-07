import os
import skimage 
from skimage import io
from skimage import transform
from glob import glob
import numpy as np
import pandas as pd
from read_dataset import build_df
from utils import CFG
from sklearn.model_selection import train_test_split
import torch
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import cv2 
from chitra.image import Chitra
from datapreprocess import Custom_Dataset , split_dataset



# Parameters
params = {'batch_size': 64,
          'shuffle': True,
          'num_workers': 6}
max_epochs = 100


# Plot numpy array
def plot_image(image):
    plt.imshow(image) # img.permute(1, 2, 0)
    plt.title(image.shape)
    plt.show()

def class_to_color(class_id):
    colors = [(255,0,0),(0,255,0),(0,0,255),(255,255,0),(255,0,255),(0,255,255),(255,100,100),
              (100,255,100),(100,100,255),(255,100,0),(255,0,100),(100,0,255),(100,100,255),(100,255,0),
              (100,255,100)]
    return colors[class_id]

# draw a single bounding box onto a numpy array image
def draw_bounding_box(img, annotation,class_id):
    # if annotation.isnull().values.any():
    #     return
    
    x_min, y_min,x_max, y_max = int(annotation[0]), int(annotation[1]),int(annotation[2]),int(annotation[3])
    
    class_id = int(class_id)
    color = class_to_color(class_id)
    img = img.transpose(1, 2, 0)
    cv2.rectangle(img,(x_min,y_min),(x_max,y_max), color, 2)
    plot_image(img)

# def train(data_loader, model, optimizer,epochs=100):
#     ''' 
#         sigma:  (num_samples,num_mixtures,2,2) 
#         pi:     (num_samples,num_mixtures)
#         mue:    (num_samples,num_mixtures,2)

#         The last parameter '2' represents x and y  
#     '''
#     num_batches = len(data_loader)
#     total_loss = 0
#     model.train()
    
#     for i in range (epochs):
#         total_loss = 0
#         for X, y in data_loader:
#             #print(X.shape)
#             x_variable = Variable(X,requires_grad=True).to(device)
#             X = X.to(device)
#             #y = y.to(device)
#             pi, sigma_x,sigma_y, mu_x , mu_y = model(x_variable)
#             #pi_variable, sigma_variable, mu_variable = model(x_variable)
#             #print(f"sigma_variable{sigma_variable.shape}")
#             loss = mdn_loss_fn( pi, sigma_x,sigma_y, mu_x , mu_y, y)
#             # loss = Variable(loss, requires_grad = True)
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()

#             total_loss += loss.item()

#         avg_loss = total_loss / num_batches
#         if (i+1)%5 == 0:
#             print(f"Epoch {i+1} train loss: {avg_loss}")
#     print(f"Epoch {i+1} train loss: {avg_loss}")

if __name__ == '__main__':
    # ['id', 'label', 'xmin', 'ymin', 'xmax', 'ymax', 'img_path']
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")

    IMG_FILES = glob(CFG.img_path + "/*.jpg")
    XML_FILES = glob(CFG.xml_path + "/*.xml")
    df, classes = build_df(XML_FILES)
    data = df.to_numpy()

    # input and target 
    input  = df[['img_path']].values
    input =np.squeeze(input)
    # input = input.reset_index()
    target = df[['xmin','ymin', 'xmax', 'ymax','label']].values.astype(np.int64)
   

    # count by groups around 1k for each class
    # group_df = df.groupby('label',sort=True).count()
    # print("group_df: ",group_df)

    # splitting data
    #train_data, validation_data, test_data = split_dataset(input,target,True)
    train_data, test_data = split_dataset(input,target,False)

    # Create dataset
    train_dataset = Custom_Dataset(train_data)
    test_dataset = Custom_Dataset(test_data)

    # sample

    #sample input
    sample_input = ((train_dataset[100])[0]).numpy()
    sample_original,sample_label  = (train_dataset.original_img(100))
    sample_class = train_dataset.get_class_label(100)
    resized = train_dataset.resized_bbox(100)
    # print(type(sample_label),sample_label)
    draw_bounding_box(np.array(sample_original),np.array(sample_label),sample_class)
    draw_bounding_box(sample_input,resized,sample_class)

    # print(train_data[0],train_data[1][0].ravel())
    
    # Dataloaders
    # train_dataloader = torch.utils.data.DataLoader(train_dataset, params['batch_size'],num_workers=params['num_workers'])
    # #validation_dataloader = torch.utils.data.DataLoader(validation_data, params['batch_size'],num_workers=params['num_workers'])
    # test_dataloader = torch.utils.data.DataLoader(test_data, params['batch_size'],num_workers=params['num_workers'])


# def show_image(array,bbox):
    