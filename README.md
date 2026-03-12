# YOLO Object Detection Project

## Project Overview

This project implements an object detection system using the YOLOv8 deep learning model.
The goal was to train a model capable of recognizing three everyday objects placed on a table:

- Red cup
- Blue bottle
- Phone

The model was trained on a custom dataset collected from photos and images downloaded from the internet. The system can detect these objects both in images and in real-time using a webcam.

---

# Dataset

The dataset was created specifically for this project. Images were collected using:

- smartphone camera
- publicly available images from the internet

The dataset contains images of:

- **red cups**
- **blue bottles**
- **phones**

Images were captured from different:

- angles
- lighting conditions
- distances
- backgrounds

Some images contain multiple objects in the same scene to make detection more realistic.

### Dataset size

Total dataset size: ~700 images

- Training images: ~600
- Validation images: ~100

### Dataset structure

```
dataset/
│
├── train/
│   ├── images/
│   └── labels/
│
├── valid/
│   ├── images/
│   └── labels/
│
└── data.yaml
```

Annotations were created using a bounding box annotation tool and exported in **YOLO format**.

---

# Model

The model was trained using the **YOLOv8 implementation from Ultralytics** with pretrained weights.

Model used:

```
yolov8n.pt
```

This is the smallest YOLOv8 model, which allows fast training even on CPU.

---

# Training

Training was performed using Python and the Ultralytics YOLO library.

Example training script:

```python
from ultralytics import YOLO

model = YOLO("yolov8n.pt")

model.train(
    data="data.yaml",
    epochs=50,
    imgsz=640
)
```

Training parameters:

- Epochs: **50**
- Image size: **640x640**
- Pretrained weights: **YOLOv8n**

---

# Results

Validation performance after training on the expanded dataset (~700 images).

Metrics are reported using the best validation checkpoint (`best.pt`).

| Metric    | Value     |
| --------- | --------- |
| Precision | 0.715     |
| Recall    | 0.715     |
| mAP50     | **0.732** |
| mAP50-95  | **0.562** |

### Performance per class

| Class       | mAP50     |
| ----------- | --------- |
| red_cup     | **0.964** |
| phone       | **0.855** |
| blue_bottle | 0.378     |

The model performed best on **red_cup**, while **blue_bottle** showed lower accuracy due to greater visual variability.

---

## Training Results

Training performance over 50 epochs.

![Training Results](example_results/training_results.png)

# Example Predictions

The trained model was evaluated on unseen images.

Example command:

```python
model.predict(
    source="test_images",
    save=True,
    conf=0.25
)
```

Predictions include:

- bounding boxes
- class labels
- confidence scores

---

Example detections on unseen images.

![Detection Example](example_results/detection_1.jpg)

![Detection Example](example_results/detection_2.jpg)

# Real-Time Detection

The model can also run real-time detection using a webcam.

Pretrained model is available in the `models/` directory.

Example script:

```python
from ultralytics import YOLO

model = YOLO("models/best.pt")

model.predict(
    source=0,
    show=True,
    conf=0.25
)
```

This demonstrates YOLO's ability to perform object detection in real time.

---

## Real-Time Detection

Example of real-time object detection using webcam.

![Webcam Detection](example_results/Webcam_detection.png)

# Technologies Used

- Python
- YOLOv8 (Ultralytics)
- PyTorch
- OpenCV
- Roboflow / Labeling tools

---

# Project Structure

```
YOLO-project/

dataset/
example_results/
models/
test_images/

README.md
data.yaml
requirements.txt

test_model.py
training_script.py
webcam_detection.py

```

---

# Future Improvements

Possible improvements include:

- increasing dataset size
- improving class balance
- adding more object variations
- training for more epochs
- using a GPU for faster training

---

## Installation

Clone the repository:

```bash
git clone https://github.com/JaneKorn/YOLO_project
```

Install dependencies:

```bash
pip install -r requirements.txt
```
