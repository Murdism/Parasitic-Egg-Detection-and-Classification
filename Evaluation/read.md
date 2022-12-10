## Results 
Results of the trained model can be found [Here](yolov7/runs/train/yolov7_cell_detection_fixed_res23/weights) 

## Inference 	
python detect.py --source ../../inference_data/video.mp4 --weights yolov7/runs/train/yolov7_cell_detection_fixed_res23/weights/best.pt --view-img
