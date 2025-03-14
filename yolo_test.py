from ultralytics import YOLO
import cv2
import os

# Load YOLOv8 model (pretrained)
model = YOLO("best-colab.onnx")  # You can also try yolov8s.pt for better accuracy

def detect_bullseye(image_path):
    # Load image
    frame = cv2.imread(image_path)
    if frame is None:
        print("Error: Image not found!")
        return

    # Resize for consistency
    frame = cv2.resize(frame, (640, 640))

    # Run YOLO detection
    results = model.predict(frame, imgsz=160)

    # Parse results
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Get bounding box
            conf = float(box.conf[0])  # Confidence score
            cls = int(box.cls[0])  # Class ID

            # If confident detection, draw box
            if conf > 0.4:  # Adjust confidence threshold
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"ID: {cls} {conf:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Show the output
    cv2.imshow("YOLO Bullseye Detection", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Run detection
# Test with multiple images
for img in os.listdir("examples"):
    detect_bullseye(f"examples/{img}")