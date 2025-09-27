# 🪖 Helmet Detection with YOLOv8

A computer vision project that detects whether people are wearing helmets using **YOLOv8**.  
This repo includes modular training & inference scripts and a **Streamlit web app** for deployment.

---

## 📂 Project Structure
```
Helmet_Detection/
│
├── models/
│ └── best.pt # trained YOLOv8 weights (not tracked in git)
│
├── scripts/
│ ├── train.py # training script
│ ├── detect.py # inference script
│ └── utils.py # helper functions
│
├── app.py # Streamlit deployment app
├── requirements.txt # dependencies
├── .gitignore # ignore cache, runs, weights
├── data/
│ └── data.yaml # dataset config
└── README.md
```
---
## 📦 Dataset

This project uses the **Bike Helmet Detection** dataset from Roboflow Universe.  
You can access it here: [Bike Helmet Detection Dataset](https://universe.roboflow.com/bike-helmets/bike-helmet-detection-2vdjo)

- Contains images of riders **with helmets** and **without helmets**.  
- Already annotated in YOLO format.  
- Used for training, validation, and testing in this project.  

⚠️ Note: The dataset itself is **not included in this repository** due to size and licensing.  
To train or fine-tune the model yourself, please download it from the above link and update the paths in `data/data.yaml`.  

---

## ⚙️ Setup

### 1️⃣ Clone the repository
```bash
git clone https://github.com/Md-Faiz-Alam/helmet_detection.git
cd Helmet_Detection
```

### 2️⃣ Create virtual environment (Python 3.13 recommended)
```
python -m venv .venv
source .venv/bin/activate   # Mac/Linux
.venv\Scripts\activate      # Windows
```

### 3️⃣ Install dependencies
```
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

### Train YOLOv8 on your dataset:
```
python scripts/train.py \
    --data data/data.yaml \
    --epochs 50 \
    --img 640 \
    --weights yolov8s.pt
```

### 🔍 Inference
Run inference on test images:
```
python scripts/detect.py \
    --weights models/best.pt \
    --source data/images/test \
    --n_samples 5
```
---

### 🌐 Streamlit App

Launch the app:
```
streamlit run app.py
```
Features:
 - Upload images or videos
 - Get real-time detection results
 - Download processed outputs

---

## 🚀 Deployment

You can deploy the app to:

- [**Streamlit Cloud**](https://docs.streamlit.io/streamlit-community-cloud)  
- [**HuggingFace Spaces**](https://huggingface.co/spaces) (Gradio/Streamlit)  
- [**Docker**](https://docs.docker.com/get-started/) for on-prem servers  

---

## 🙌 Acknowledgements

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)  
- [Roboflow](https://roboflow.com/) for dataset management  
