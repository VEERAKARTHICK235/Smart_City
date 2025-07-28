import random

def detect_objects_mock(image_path):
    """Mocks the output of a YOLOv8 object detection model."""
    print(f"🚓 Analyzing image: {image_path}")
    
    # In a real scenario, you'd use a library like 'ultralytics'
    # from ultralytics import YOLO
    # model = YOLO('yolov8n.pt')
    # results = model(image_path)
    # But here, we simulate the results.
    
    objects = ['person', 'car', 'bicycle', 'traffic light']
    detected_objects = []
    num_detections = random.randint(1, 5)

    for _ in range(num_detections):
        obj = {
            "label": random.choice(objects),
            "confidence": random.uniform(0.65, 0.98),
            "box": [random.randint(100, 800) for _ in range(4)] # [x1, y1, x2, y2]
        }
        detected_objects.append(obj)
        
    return detected_objects

# Example Usage:
# detections = detect_objects_mock('path/to/cctv_image.jpg')
# print(f"Detected {len(detections)} objects.")
# for d in detections:
#     print(f" - {d['label']} with {d['confidence']:.2f} confidence.")