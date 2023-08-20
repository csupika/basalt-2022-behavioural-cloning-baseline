from ultralytics import YOLO

original = "~/data/yolov5/MineRLInsideCave_v1"
no_water = "~/data/yolov5/MineRLInsideCave_v2_nowater"
train_no = [(18, original),(19,original),(22,no_water),(23,no_water)]

for no, data in train_no:
    print(f"\n{no}")
    # Load a model
    model = YOLO(f'runs/classify/train{no}/weights/last.pt')

    # Validate the model
    metrics = model.val(data=data)  # no arguments needed, dataset and settings remembered

