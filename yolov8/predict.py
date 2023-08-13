from ultralytics import YOLO
import numpy as np


model = YOLO('./runs/classify/train2/weights/best.pt')  # load a custom model
results = model('../data/yolov5/recording_frames_1st_sample/cheeky-cornflower-setter-0b539fe8872c-20220713-175120.mp4_frame_1673.jpg', verbose=True, conf=0.8)  # predict on an image

# Get the names and probabilities
names_dict = results[0].names
probs = results[0].probs.data.tolist()

print(names_dict[np.argmax(probs)] == 'inside cave')
print(probs[1])

