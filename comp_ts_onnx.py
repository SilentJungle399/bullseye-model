from ultralytics import YOLO
import os

# Load the exported ONNX model
best_colab_onnx_model = YOLO("best-130.onnx")

for i in os.listdir("examples"):
	img_path = f"examples/{i}"
    
	# ONNX model
	# results = best_onnx_model.predict(img_path, imgsz = 320)

	# Torchscript model
	# results = best_ts_model.predict(img_path, imgsz = 320)

	# ONNX model
	results = best_colab_onnx_model.predict(img_path, imgsz = 160)
	for result in results:
		result.show()

	# Torchscript model
	# results = best_colab_ts_model.predict(img_path, imgsz = 320)
