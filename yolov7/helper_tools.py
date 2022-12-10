import os
from glob import glob
import numpy as np
# import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch.autograd import Variable
from tqdm import tqdm
from datapreprocess import preprocess_image
import seaborn as sns
from sklearn.metrics import *
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
import torchvision
import torchvision.transforms as transforms
from skimage import io
import skimage
from PIL import Image
from chitra.image import Chitra

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, data, transform=None,IMAGE_SIZE = 320):
        super(CustomDataset, self).__init__()
        # List of files
        self.IMAGE_SIZE = IMAGE_SIZE
        self.data_files = data[0]  
        self.label_files = data[1]
        self.class_labels = torch.LongTensor(self.get_class_labels()) 
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
    
    def get_class_labels(self):
        labels  = []
        for fname in self.label_files:
            with open(fname, "r") as file:
                class_label  = int(file.read().split(" ")[0])
                labels.append(class_label)
        return np.array(labels,dtype=np.uint8)

    def original_img(self,idx):
        self.img =(1/255
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
        
        if self.transform is not None:

            img  = np.asarray(io.imread(self.data_files[idx], plugin="pil")) # numpy image
            pil_img  = Image.fromarray(img)
            data_p = self.transform()(pil_img) # [C, H, W] format image is returned.
            label_p = self.class_labels[idx]

            return data_p,label_p

        else:
          
            self.data = (
                (1/255 )
                *np.asarray(
                io.imread(self.data_files[idx], plugin="pil").transpose(
                    (2, 0, 1)
                ),
                dtype="float32",
            ))
            data_p= self.data
            # Return the torch.Tensor values
            return torch.from_numpy(data_p), self.class_labels[idx]

class CustomDatasetV2(torch.utils.data.Dataset):
    def __init__(self, data, transform_fn,IMAGE_SIZE = 320):
        super(CustomDatasetV2, self).__init__()
        # List of files
        self.IMAGE_SIZE = IMAGE_SIZE
        self.data_files = data[0]  
        self.label_files = data[1]
        self.class_labels = torch.LongTensor(self.get_class_labels()) 
        self.transform = transform_fn()
        # Sanity check : raise an error if some files do not exist
        for f in self.data_files:
            if not os.path.isfile(f):
                raise KeyError("{} is not a file !".format(f))

    def __len__(self):
        return len(self.data_files)  # the length of the used data
    
    def img_size(self,idx):
        self.img =np.asarray(io.imread(self.data_files[idx]))
        return (self.img).shape
    
    def get_class_labels(self):
        labels  = []
        for fname in self.label_files:
            with open(fname, "r") as file:
                class_label  = int(file.read().split(" ")[0])
                labels.append(class_label)
        return np.array(labels,dtype=np.uint8)

    def original_img(self,idx):
        self.img =(1/255
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

            # preprocessing steps to resnet format image

            img  = np.asarray(io.imread(self.data_files[idx], plugin="pil")) # numpy image
            pil_img  = Image.fromarray(img) # convert the numpy image to PIL Image
            resnet_data_p = self.transform(pil_img) # transform PIL image to [C, H, W] formatted pytorch tensor
            label_p = self.class_labels[idx]

            return resnet_data_p,label_p






def reader(IMG_FILES,LABEL_FILES):
    # x_train, x_valid, x_test, y_train, y_valid, y_test
    dir_names = ['train','valid','test']
    # images
    x_train, x_valid, x_test, y_train, y_valid, y_test=[],[],[],[],[],[]

    for dir in os.listdir(IMG_FILES):
        if dir in dir_names:
            for img_name in os.listdir(os.path.join(IMG_FILES, dir)):
                img_path = os.path.join(IMG_FILES, dir, img_name)

                if dir  == 'train':
                    x_train.append(img_path)

                elif dir == 'valid':
                    x_valid.append(img_path)

                elif dir  == 'test':
                    x_test .append(img_path)
    
    for l_dir in os.listdir(LABEL_FILES):
        if l_dir in dir_names:
            for label in (os.listdir(os.path.join(LABEL_FILES, l_dir))):
                label_path = os.path.join(LABEL_FILES, l_dir,label)

                if l_dir  == 'train':
                    y_train.append(label_path)

                elif l_dir == 'valid':
                    y_valid.append(label_path)

                elif l_dir  == 'test':
                    y_test .append(label_path)




    train_data = [sorted(x_train),sorted(y_train)]
    valid_data = [sorted(x_valid),sorted(y_valid)]
    test_data = [sorted(x_test),sorted(y_test)]
    return np.array(train_data, dtype=object), np.array(valid_data, dtype=object), np.array(test_data, dtype=object)

    # class labels

def get_default_device():
    """gets gpu for mac m1 or cuda, or cpu machine"""
    
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print('Running on Cuda GPU')
        return device
        # x = torch.ones(1, device=mps_device)
        # print(x)
        
    
    elif torch.backends.mps.is_available():
        print('Running on the Mac GPU')
        mps_device = torch.device("mps")
        return mps_device
        
    else:
        # print("MPS device not found.")
        return torch.device('cpu')





class TinyDataset(torch.utils.data.Dataset):
    def __init__(self,data):
        super(TinyDataset,self).__init__()
        self.X  = data[0]
        self.y = torch.LongTensor(data[1])

    def __getitem__(self,idx):
        return self.X[idx], self.y[idx]

    def __len__(self):
        return len(self.y)



def plot_model_history(training_accs, training_losses):# Get training and test loss histories

    # Create count of the number of epochs
    epoch_count = range(1, len(training_accs) + 1)

    # Visualize loss history
    plt.plot(epoch_count, training_losses, 'r--')
    plt.plot(epoch_count, training_accs, 'b-')
    plt.legend(['Training Loss', 'Training Accuracy'])
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy and Loss Curves')
    plt.show()

# testing/ validation of unet model
def validate_model(trained_model, test_dataloader, device=None):
    trained_model.eval() # switch off some layers such as dropout layer during validation
    with torch.no_grad():
        test_predictions = []
        test_labels = []

        for i, (x, y) in enumerate(test_dataloader):
            x  = x.to(device)
            y  = y.to(device)
            y_hat = trained_model(x)
            predicted_labels =  torch.argmax(y_hat, dim=1).detach().cpu().numpy().tolist()
            # y_hat  = y_hat.cpu()
            y = y.detach().cpu().numpy().tolist()
            test_predictions.extend(predicted_labels)
            test_labels.extend(y)
    
        return test_labels, test_predictions


def evalution_metrics(ground_truth, predictions):
    print(f"mean acc score = {accuracy_score(ground_truth, predictions)}")
    print(f"mean recall score = {recall_score(ground_truth, predictions, average='micro')}")
    print(f"precision score = {precision_score(ground_truth, predictions, average='micro')}")
    print(f"mean f1 score = {f1_score(ground_truth, predictions, average='micro')}")
    labels = np.unique(ground_truth).tolist()
    cm  = confusion_matrix(ground_truth, predictions, labels=labels) 
    report  = classification_report(ground_truth, predictions)
    print(report)

    sns.heatmap(cm)
    plt.show()



def get_pretrained_model():
    
    resnet152 = torchvision.models.resnet152(pretrained=True)
    resnet152.eval()
    # remove the last layer
    resnet152_model = torch.nn.Sequential(*(list(resnet152.children())[:-1]))
    # freeze the model
    for param in resnet152_model.parameters():
        param.requires_grad = False

    return resnet152_model



def trainCustomModel(resnetModel,yoloDetectModel,custom_cnn_classifier, train_dataloader,optimizer,loss_fn, epochs=30, learning_rate=0.001, device='cpu'):
    """Accepts feature from resnet 
    and yolo object detection cropped iamge(s) 
    as features to train an accurate cnn classifier.
    """

    training_losses = []
    training_accs = []

    for epoch in range(1, epochs+1):
        number_of_batches = 0
        epoch_loss_values = 0.0
        epoch_accs = 0.0
        for index, (X1, y) in enumerate(tqdm(train_dataloader)):
            # put tensors in gpu state
            X1  = Variable(X1, requires_grad=True).to(device)
            # X2 = Variable(X2, requires_grad=True).to(device)
            y  = y.to(device)

            # resnet processing
            # yolo_X  = yoloDetectModel(X1)

            resnet_X  = resnetModel(X1)

            # extracted_features = resnetModel(resnet_X) # (Bacth, 1000) tensor is returned
            # X2  = Variable(torch.randn(size= (X.size()[0],1000)), requires_grad=True).to(device)
            preds = custom_cnn_classifier(X1, resnet_X)
        

            loss = loss_fn(preds, y).to(device)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            probs = torch.log_softmax(preds, dim=1)
            predicted_labels = torch.argmax(probs, dim=1)

            # acc
            epoch_accs += accuracy_score(y.detach().cpu(),predicted_labels.detach().cpu())
            epoch_loss_values += loss.item()

            number_of_batches += 1

        # compute average of batch loss and accuracy
        batch_acc, batch_loss = epoch_accs / \
            number_of_batches, epoch_loss_values / number_of_batches
        training_losses.append(batch_loss)
        training_accs.append(batch_acc)

        print("Epoch:{}/{}, acc={:.3f}%, loss={:.3f}".format(epoch,epochs, batch_acc*100, batch_loss))

    print("Learning Finished!")

    return custom_cnn_classifier, training_accs, training_losses