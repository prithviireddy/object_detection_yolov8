
---

# **Dog Breed Detection Using YOLOv8**

A custom object detection project trained to recognize 18 Indian dog breeds using YOLOv8.
This repository contains the training scripts, evaluation code, and inference pipeline for running the model locally with GPU acceleration.

---

# **1. Setup Instructions**

This section walks through setting up CUDA, PyTorch, and the required dependencies to run the project on your local machine.

---

## **1.1 Install CUDA (GPU Acceleration)**

1. Check your NVIDIA GPU driver:

   ```
   nvidia-smi
   ```
2. Install the CUDA Toolkit version compatible with your GPU:
   [https://developer.nvidia.com/cuda-downloads](https://developer.nvidia.com/cuda-downloads)
3. Install cuDNN (matching your CUDA version):
   [https://developer.nvidia.com/cudnn](https://developer.nvidia.com/cudnn)

**Important:**
CUDA, cuDNN, and PyTorch must **all match versions**, otherwise training will throw errors.

---

## **1.2 Install PyTorch (with CUDA support)**

After installing CUDA, install the matching PyTorch build.
Get your exact command from:
[https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/)

Example (CUDA 12.1):

```
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

Verify installation:

```python
import torch
torch.cuda.is_available()   # should return True
torch.cuda.get_device_name() 
```

---

## **1.3 Clone This Repository**

```
git clone https://github.com/yourusername/dog-breed-detection-yolov8.git
cd dog-breed-detection-yolov8
```

---

## **1.4 Install Python Dependencies**

After cloning, install required packages:

```
pip install -r requirements.txt
```

Typical requirements include:

* ultralytics
* opencv-python
* numpy
* matplotlib

---

# **2. Dataset Setup**

This project uses a dataset created and labeled using **Roboflow**, exported in YOLOv8 format.

### **Steps**

1. Download your Roboflow dataset (YOLOv8 version).
2. Extract it to your system.
3. Ensure your folder structure looks like:

```
dataset/
   train/images/
   train/labels/
   val/images/
   val/labels/
   test/images/
data.yaml
```

`data.yaml` should contain:

* path to dataset
* train/val/test splits
* list of class names (18 dog breeds)

---

# **3. Training the Model**

Use the YOLOv8n base model to start training:

```python
from ultralytics import YOLO

DATA_YAML = r"path_to_data.yaml"
model = YOLO("yolov8n.pt")

model.train(
    data=DATA_YAML,
    epochs=20,
    imgsz=640,
    batch=16,
    workers=6,
    device=0,
    name="yolo_final",
    cache="disk",
    augment=True,
)
```

### **Outputs Saved Automatically**

YOLO saves:

```
runs/detect/yolo_final/
    weights/
        best.pt
        last.pt
    results.png
    confusion_matrix.png
```

---

# **4. Model Evaluation**

Load the best-performing model and run validation:

```python
from ultralytics import YOLO

best = r"runs/detect/yolo_final/weights/best.pt"
model = YOLO(best)

metrics = model.val()
```

This prints:

* mAP@50
* mAP@50-95
* Precision
* Recall

---

# **5. Running Predictions**

Place your test images here:

```
dataset/test/newimgs/
```

Run inference:

```python
model.predict(source="dataset/test/newimgs", save=True, imgsz=640)
```

Results will be saved to:

```
runs/detect/predict/
```

YOLO automatically draws bounding boxes and labels on the output images.

---

# **6. Project Structure**

```
├── data.yaml
├── notebooks/
│   ├── training.ipynb
│   ├── evaluation.ipynb
│   └── inference.ipynb
├── runs/
│   └── detect/
├── dataset/
├── main.py
└── README.md
```

---

# **7. References**

* YOLOv8 Documentation: [https://docs.ultralytics.com](https://docs.ultralytics.com)
* PyTorch Installation Guide: [https://pytorch.org](https://pytorch.org)
* Roboflow Dataset: [https://roboflow.com](https://universe.roboflow.com/dogs-classifier/indian-dog-breeds/dataset/3/download/yolov8)

---
