from PIL import Image
from torchvision import transforms
import torch
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
        return self.img,self.labels[idx]

    def get_class_label(self,idx):
        return self.class_labels[idx]

    def resized_bbox(self,idx):
        image = Chitra(self.data_files[idx], self.labels[idx], self.class_labels[idx])
        # Chitra can rescale your bounding box automatically based on the new image size.
        image.resize_image_with_bbox((640, 640))

        return np.array([image.bboxes[0].x1_int,image.bboxes[0].y1_int,image.bboxes[0].x2_int,image.bboxes[0].y2_int])

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
        validation_data = [X_valid, y_valid]
        return train_data, validation_data, test_data

    train_data = [X_train, y_train]
    test_data = [X_test, y_test]

    return train_data, test_data
def preprocess_image():
    """make the images into restnet 50 model format and normalization"""
    preprocess = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ]
    )
    return preprocess


if __name__ == "__main__":
    test_tensor = torch.randint(0, 255, size=(3, 200, 200), dtype=torch.uint8)

    tensor2image_transform = transforms.ToPILImage()
    img = tensor2image_transform(test_tensor)
    input_tensor = preprocess_image()(img)
    print(input_tensor.shape)
