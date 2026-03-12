from ultralytics import YOLO

model = YOLO("models/best.pt")

results = model.predict(
    source="test_images",
    save=True,
    conf=0.25
)