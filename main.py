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
    print(x_min, y_min,x_max, y_max)
    print("The shape: ",img.shape)
    img = img.transpose(1, 2, 0)
    print(img)
    cv2.rectangle(img,(x_min,y_min),(x_max,y_max), color, 2)
    plot_image(img)
class LoadDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        super(LoadDataset, self).__init__()
        # List of files
        self.data_files = data[0]  # [DATA_FOLDER.format(id) for id in ids]
        self.labels = data[1]  # [LABELS_FOLDER.format(id) for id in ids]
        # Sanity check : raise an error if some files do not exist
        for f in self.data_files:
            if not os.path.isfile(f):
                raise KeyError("{} is not a file !".format(f))

    def __len__(self):
        return len(self.data_files)  # the length of the used data
    
    def img_size(self,idx):
        self.img =np.asarray(io.imread(self.data_files[idx]))
        return (self.img).shape

    def __getitem__(self, idx):
        #         Pre-processing steps
        #     # Data is normalized in [0, 1]
        
        self.data = (
    np.asarray(
                io.imread(self.data_files[idx], plugin="pil")
            )
        )
        # print("label = ", self.labels.shape)
        # draw_bounding_box(self.data, self.labels[idx],self.labels[idx][4])
        data_p, label_p = self.data, self.labels[idx]
        # Return the torch.Tensor values
        return torch.from_numpy(data_p), torch.from_numpy(label_p)



class Custom_Dataset(torch.utils.data.Dataset):
    def __init__(self, data, transform=None,IMAGE_SIZE = 640):
        super(Custom_Dataset, self).__init__()
        # List of files
        self.IMAGE_SIZE = IMAGE_SIZE
        self.data_files = data[0]  # [DATA_FOLDER.format(id) for id in ids]
        self.labels = data[1][:,:4]  # [LABELS_FOLDER.format(id) for id in ids]
        self.class_labels = torch.LongTensor(data[1][:,4]) 
        self.transform = transform
        # Sanity check : raise an error if some files do not exist
        for f in self.data_files:
            if not os.path.isfile(f):
                raise KeyError("{} is not a file !".format(f))

    def __len__(self):
        return len(self.data_files)  # the length of the used data
    
    def img_size(self,idx):
        self.img =np.asarray(io.imread(self.data_files[idx]))
        return (self.img).shape
    def original_img(self,idx):
        self.img =(
                   1/255
             *np.asarray(
                io.imread(self.data_files[idx], plugin="pil").transpose(
                    (2, 0, 1)
                ),
                dtype="float32",
            )
        )
        return (self.img)

    def get_class_label(self,idx):
        return self.class_labels[idx]

    def resized_bbox(self,idx):
        image = Chitra(self.data_files[idx], self.labels[idx], self.class_labels[idx])
        # Chitra can rescale your bounding box automatically based on the new image size.
        image.resize_image_with_bbox((640, 640))

        return [image.bboxes[0].x1_int,image.bboxes[0].y1_int,image.bboxes[0].x2_int,image.bboxes[0].y2_int]

    def __getitem__(self, idx):
        #         Pre-processing steps
        #     # Data is normalized in [0, 1]
        
 
        if self.transform is not None:

            self.data = (
                np.asarray(
                io.imread(self.data_files[idx], plugin="pil").transpose(
                    (2, 0, 1)
                ),
                dtype="float32",
            )
        )

            # convert the tensor to image
            tensor2image_transform = transforms.ToPILImage()

            # make the format avialble to resnet model
            data_p = torch.tensor(
                self.data 
            )  # tranform the image into ResNet format

            image = tensor2image_transform(data_p)
            data_p = self.transform()(
                image
            )  # transform the image into ResNet format
             
            label_p = self.class_labels[idx]
            # print(self.labels[idx][4])

            return data_p,label_p
        else:   
            img = io.imread(self.data_files[idx], plugin="pil")
            resized_img = skimage.transform.resize(img, (self.IMAGE_SIZE, self.IMAGE_SIZE))
            self.data = (
     
             np.asarray(
               resized_img.transpose(
                    (2, 0, 1)
                ),
                dtype="float32",
            )
            )
            # print("self.data_files[idx] : " ,self.data_files[idx])
            # print("label = ", self.label.shape).
            label_p = self.resized_bbox(idx)
            data_p= self.data
            # Return the torch.Tensor values
            return torch.from_numpy(data_p), torch.from_numpy(label_p)



def split_dataset(train,target,validation=False):
    #70%-20%-10% split, as we're splitting 10% from the already split X_train so we're actually ending up with a 72%-20%-8% split here:
    #80 -20
    # x = img_path
    # y = 'xmin', 'ymin', 'xmax', 'ymax', 'label'

    X_train, X_test, y_train, y_test = train_test_split(
        train, target, train_size=0.8, shuffle=True, stratify=target[:, 4]
    )

    if validation:
        X_train, X_valid, y_train, y_valid = train_test_split(
            X_train,
            y_train,
            train_size=0.9,
            shuffle=True,
            stratify=target[:, 4],
        )
        train_data = [X_train, y_train]
        validation_data = [X_train, y_valid]
        return train_data, validation_data, test_data

    train_data = [X_train, y_train]
    test_data = [X_test, y_test]

    return train_data, test_data


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
    sample_original = train_dataset.original_img(100)
    sample_label = (train_dataset[100])[1] 
    sample_class = train_dataset.get_class_label(100)
    resized = train_dataset.resized_bbox(100)
    print(type(resized))
    draw_bounding_box(sample_original,sample_label,sample_class)
    draw_bounding_box(sample_input,resized,sample_class)

    # print(train_data[0],train_data[1][0].ravel())
    
    # Dataloaders
    # train_dataloader = torch.utils.data.DataLoader(train_dataset, params['batch_size'],num_workers=params['num_workers'])
    # #validation_dataloader = torch.utils.data.DataLoader(validation_data, params['batch_size'],num_workers=params['num_workers'])
    # test_dataloader = torch.utils.data.DataLoader(test_data, params['batch_size'],num_workers=params['num_workers'])


# def show_image(array,bbox):
    