from ultralytics import YOLO

model = YOLO("best-130.pt")  # Load your model


model.export(format = "ONNX", imgsz = 160)  # Export to ONNX