from ultralytics import YOLO

model = YOLO("runs/detect/train5/weights/best.pt")

model.predict(source=0, show=True)