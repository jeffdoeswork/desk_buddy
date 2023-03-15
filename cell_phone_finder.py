import torch
import yolov5

# Load YOLOv5 model
model = yolov5.load('yolov5s.pt')

# Load image
img = 'path/to/image.jpg'

# Perform detection
results = model(img)

# Print bounding box coordinates of detected objects
for detection in results.xyxy[0]:
    x1, y1, x2, y2, conf, cls = detection.tolist()
    print(f'Object {int(cls)} detected with confidence {conf:.2f} at coordinates ({x1:.2f}, {y1:.2f}), ({x2:.2f}, {y2:.2f})')
