from ultralytics import YOLO

if __name__ == "__main__":
	# Load a model
	model = YOLO(r"yolov5nu.yaml")  # load last trained state
	# model = YOLO(r"runs\detect\train14\weights\last.pt")  # load last trained state

	# Train the model
	results = model.train(
		data = "data-win.yaml", 
		name = "bullseye-model",
		epochs = 200, 
		imgsz = 160, 
		batch = 0.6, 
		plots = True, 
		lrf = 0.01,
		# pretrained = True,
		resume = True # resume training from the last epoch
	)
