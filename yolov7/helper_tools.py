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
import argparse
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
import numpy as np

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel

# files = ['cell_datatset/images/train/Ascaris lumbricoides_0006.jpg','cell_datatset/images/train/Trichuris trichiura_0992.jpg']
# images = detect(files)


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

def detect(files=None,Not_path = True,save_img=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='runs/train/yolov7_cell_detection_fixed_res23/weights/best.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='cell_datatset/images/valid', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--no-trace', action='store_true', help='don`t trace model')
    opt = parser.parse_args()
    source, weights, view_img, save_txt, imgsz, trace = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size, not opt.no_trace
    if files is not None:
        source = files


    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size

    # if trace:
    #     model = TracedModel(model, device, opt.img_size)

    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

    # Set Dataloader
    dataset = LoadImages(source, img_size=imgsz, stride=stride,Not_path=Not_path)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    old_img_w = old_img_h = imgsz
    old_img_b = 1

    images_all =[]
    counter = 0

    t0 = time.time()
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Warmup
        if device.type != 'cpu' and (old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
            old_img_b = img.shape[0]
            old_img_h = img.shape[2]
            old_img_w = img.shape[3]
            for i in range(3):
                model(img, augment=opt.augment)[0]

        # Inference
        t1 = time_synchronized()
        with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
            pred = model(img, augment=opt.augment)[0]
        t2 = time_synchronized()

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t3 = time_synchronized()

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)
        

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)


            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # # Write results
                for *xyxy, conf, cls in reversed(det):
                    #xyxy [x1, y1, x2, y2]
                        # Adding small w and height to increase the size of the images
                        w = 5
                        h = 5
                        x1,y1,x2,y2  = int(xyxy[0]) + w, int(xyxy[1]) + h, int(xyxy[2]) + w, int(xyxy[3]) + h
                        #cropped_image = img[Y:Y+H, X:X+W]
                        cropped_image = im0[y1:y2, x1:x2]
                        #cv2.imshow("cropped", cropped_image)
                        resized_image = cv2.resize(cropped_image, (224, 224))
                        images_all.append(resized_image) 
                        counter+=1
                        print('image: ',counter)



            # Print time (inference + NMS)
            print(f'{s}Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference, ({(1E3 * (t3 - t2)):.1f}ms) NMS')

    print(f'Done. ({time.time() - t0:.3f}s)')
    
    return np.array(images_all)
