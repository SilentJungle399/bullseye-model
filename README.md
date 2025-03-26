# YOLO model for Bullseye detection

This repository contains code for training a YOLO model to detect a bullseye.

## Dataset
A custom dataset was used to train the model. The dataset consists of images containing bullseyes.<br>
The dataset must follow YOLOv11 standards and must be placed inside a directory named `dataset` at the project root.

## Models
Two primary models were trained:
- **best**: Trained on my PC
- **best-colab**: Trained on Google Colab

## Files & Scripts

### Training & Model Comparison
- `train.py`: Contains all parameters to train the model using Ultralytics library.
- `comp_ts_onnx.py`: Compares the inference time of ONNX and TorchScript formats for both best and best-colab models on a Raspberry Pi 4B.

### Benchmarking
- `benchmark_model.py`: Runs benchmarks on the trained models.
- `benchmarks.log`: Contains benchmark results for both `best` and `best-colab` models.

### Testing
- `opencv_test.py`: Tests the model on a set of pre-saved bullseye images.
- `picam_test.py`: Tests the Raspberry Pi camera feed and evaluates OpenCV’s HoughCircles method for detecting bullseyes (which was found inefficient).
- `yolo_test.py`: Runs live inference using the trained YOLO model on a camera stream.

### Dataset Configuration
- `data-linux.yaml` & `data-win.yaml`: YAML files specifying dataset paths (different absolute paths for Linux and Windows).

## Results
- The [results.csv](/runs/detect/bullseye-model4/results.csv) has training results over 130 epochs.

| Epoch | Train Box Loss | Train Cls Loss | Train DFL Loss | Precision (B) | Recall (B) | mAP50 (B) | mAP50-95 (B) | Val Box Loss | Val Cls Loss | Val DFL Loss | LR PG0  | LR PG1  | LR PG2  |
| ----- | -------------- | -------------- | -------------- | ------------- | ---------- | --------- | ------------ | ------------ | ------------ | ------------ | ------- | ------- | ------- |
| 1     | 3.8923         | 3.30073        | 4.08417        | 0.01545       | 0.46829    | 0.0866    | 0.06706      | 3.62925      | 3.66435      | 4.14368      | 0.0006  | 0.0006  | 0.0006  |
| 2     | 2.87993        | 1.98997        | 3.08428        | 0.00706       | 0.51951    | 0.18509   | 0.12929      | 3.6334       | 3.48378      | 4.14384      | 0.00121 | 0.00121 | 0.00121 |
| 3     | 1.96915        | 1.49711        | 2.19448        | 0.23748       | 0.34634    | 0.15484   | 0.06874      | 2.9761       | 2.70784      | 3.03805      | 0.00182 | 0.00182 | 0.00182 |
| ..... | .............. | .............. | .............. | ............. | .......... | ........... | .......... | ............ | ............ | ............ | ....... | ....... | ....... |
| 129   | 0.80191        | 0.58796        | 1.18512        | 0.99647       | 0.97805    | 0.97559   | 0.79891      | 0.74167      | 0.3649       | 1.15386      | 0.00073 | 0.00073 | 0.00073 |
| 130   | 0.79106        | 0.58632        | 1.17945        | 0.99751       | 0.97783    | 0.97597   | 0.78646      | 0.77879      | 0.3609       | 1.16972      | 0.00072 | 0.00072 | 0.00072 |
| 131   | 0.79467        | 0.58256        | 1.17882        | 0.99669       | 0.97805    | 0.97628   | 0.79426      | 0.75459      | 0.34918      | 1.15876      | 0.00071 | 0.00071 | 0.00071 |

![results](https://github.com/SilentJungle399/bullseye-model/blob/main/results.png?raw=true)

