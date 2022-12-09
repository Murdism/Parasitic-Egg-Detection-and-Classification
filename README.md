# Parasitic-Egg-Detection-and-Classification
The main goal of this project is to detection and identify Intestinal parasite eggs.
This task was ICIP 2022 challenge problem.

## Background
More infor about the task (challenge) can be found [Here](https://icip2022challenge.piclab.ai/)


In this project we use transfer learning using yolo 7 on Chula dataset.

The link to the original yolo 7 implementation is [here](https://github.com/WongKinYiu/yolov7)

## Dataset
The Chula dataset contains around 11k images of 11 different classes of parastic eggs.
[Here](https://kaggle.com/datasets/5483e3ebb7abafb3d22876dbc921cce5adce33ffb318a6676fc39c465fff6a4b) is a link to the original dataset.

### Preprocessing source code  
Some sourcode to pre-process dataset can be found [Here](https://data.mendeley.com/v1/datasets/ytf4xwvy69/draft?a=19da38f9-4716-46fd-9715-fe368b98ba85).

### Pre-Processed Dataset
The original dataset needs pre-processing to be used yolo 7. We pre-proceesed the dataset for convinient format to be used by yolo 7.

[Link](https://drive.google.com/file/d/1bQvOkOqv5YWJPhr2f9RH-4tM7b7VKFfM/view?usp=share_link)

### Data format
After downloading the dataset the directory should look like :

|--cell_dataset <br/>
    |---images  <br/>
    |       |---train<br/>
    |       |---test<br/>
    |       |---valid <br/>
    |---labels<br/>
            |---train<br/>
            |---test<br/>
            |---valid <br/>


## Training
To train the model:
- writefile data/cell_detection.yaml
- writefile cfg/training/cell_detection.yaml

```
python train.py --epochs 10 --workers 6 --device 0 --batch-size 12 --data data/cell_detection.yaml \
--img 640 640 --cfg cfg/training/cell_detection.yaml --weights 'yolov7_training.pt' \
--name yolov7_cell_detection_fixed_res --hyp data/hyp.scratch.custom.yaml
```
## Results 
Results of the trained model can be found ./yolov7/runs/train/yolov7_cell_detection_fixed_res23

