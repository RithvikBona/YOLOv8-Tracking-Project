from ultralytics import YOLO


model = YOLO("yolov8s.pt")

## Evaluating Base YOLO on Datasetz

print("Testing pre-trained YOLO model on KITTI...")

before_metrics = model.val(data='./data.yaml', batch=8, imgsz=640)
print("Before Metrics Eval mAP50: ", before_metrics.results_dict['metrics/mAP50(B)'])


## Retrain the model
print("Retraining Model....")
model.train(data='./data.yaml', epochs=10, batch=8, imgsz=640, lr0=0.01)

## Test on Validation Set, and Display Results
print("Testing re-trained YOLO model on KITTI...")

metrics = model.val(data='./data.yaml', batch=8, imgsz=640)
print("Retrained Model Eval mAP50: ", metrics.results_dict['metrics/mAP50(B)'])