from ultralytics.utils.benchmarks import benchmark
import os, sys

models = [
	"best.pt",
]

if __name__ == "__main__":
	benchmark(
		model=models[0], 
		data="data-win.yaml" if sys.platform != "linux" else "data-linux.yaml", 
		imgsz=128, 
		half=True, 
		device= "cuda:0" if sys.platform != "linux" else "cpu",
		verbose=True,
		eps=0.001
	)
