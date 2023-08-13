from ultralytics import YOLO
train_no = [20,21]

for no in train_no:
    print(f"\n{no}")
    # Load a model
    model = YOLO(f'runs/classify/train{no}/weights/last.pt')  # load model

    # Validate the model
    metrics = model.val()  # no arguments needed, dataset and settings remembered

