from ultralytics import YOLO

model = YOLO("runs/detect/train4/weights/best.pt")

results = model.predict(
    source="test_images",
    save=True,
    conf=0.25
)