import time

from ultralytics import YOLO

# Load a model
model = YOLO('yolov8n-cls.pt')  # load a pretrained model

# Train the model
model.train(data='../data/yolov5/MineRLInsideCave_v2_nowater', patience=100, epochs=100, imgsz=640, batch=16, amp=False, device=[0,1,2,3])

