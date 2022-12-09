# Parasitic-Egg-Detection-and-Classification
Intestinal parasite egg detection and type identification

## Dataset
https://drive.google.com/file/d/1bQvOkOqv5YWJPhr2f9RH-4tM7b7VKFfM/view?usp=share_link

### finetune p5 models
python train.py --workers 8 --device 0 --batch-size 32 --data data/custom.yaml --img 640 640 --cfg cfg/training/yolov7-custom.yaml --weights 'yolov7_training.pt' --name yolov7-custom --hyp data/hyp.scratch.custom.yaml


### finetune p6 models
python train_aux.py --workers 8 --device 0 --batch-size 16 --data data/custom.yaml --img 1280 1280 --cfg cfg/training/yolov7-w6-custom.yaml --weights 'yolov7-w6_training.pt' --name yolov7-w6-custom --hyp data/hyp.scratch.custom.yaml




### Run
python train.py --epochs 10 --workers 6 --device 0 --batch-size 12 --data data/cell_detection.yaml \
--img 640 640 --cfg cfg/training/cell_detection.yaml --weights 'yolov7_training.pt' \
--name yolov7_cell_detection_fixed_res --hyp data/hyp.scratch.custom.yaml