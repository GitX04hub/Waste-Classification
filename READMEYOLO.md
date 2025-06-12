# Waste-Classification
A waste classification model for classifying images/videos into categories like glass, paper, metal, cardboard etc.  The purpose of this model is to implement deep learning and object detection techniques for classification. By leveraging these to technologies the aim is to build a enhanced model.

# ♻️ Waste Classification using YOLOv8

This project implements a deep learning-based object detection model using YOLOv8 to classify various types of waste (plastic, metal, paper, etc.) in images and videos. The model is designed to help improve waste sorting and recycling through real-time detection using a web-based interface built with Streamlit.

---
Absolutely! Here's a **detailed, step-by-step guide** you can include in your `README.md` or use to document how you implemented the **YOLOv8-based waste classification model** cloned from GitHub.

---

## 🛠️ Step-by-Step Guide: Implementing YOLOv8 Waste Classification from GitHub

---

### ✅ **Step 1: Clone the Repository**

First, clone the project repository from GitHub to your local machine.

```bash
git clone https://github.com/teamsmcorg/Waste-Classification-using-YOLOv8.git
cd Waste-Classification-using-YOLOv8
```

---

### ✅ **Step 2: Set Up a Virtual Environment (Optional but Recommended)**

Create and activate a virtual environment named `collegeP`:

```bash
python -m venv collegeP
collegeP\Scripts\activate  # Windows
# or
source collegeP/bin/activate  # macOS/Linux
```

---

### ✅ **Step 3: Install Required Dependencies**

Install all Python dependencies needed to run the YOLOv8 model and the Streamlit interface:

```bash
pip install -r requirements.txt
```

If `requirements.txt` doesn't exist, install manually:

```bash
pip install ultralytics streamlit opencv-python pafy youtube-dl
```

---

### ✅ **Step 4: Ensure Pretrained YOLOv8 Model is Available**

Place the trained YOLOv8 model (`best.pt` or `yolov8_custom.pt`) in the correct path, such as:

```
Waste-Classification-using-YOLOv8/streamlit-detection-tracking - app/weights/yolov8_custom.pt
```

If you don't have a model, download one from [Ultralytics Release Page](https://github.com/ultralytics/assets/releases) or train your own.

---

### ✅ **Step 5: Modify the Code to Load Your Model**

Open `helper.py` and update the model loading path:

```python
model = YOLO('path/to/your/yolov8_custom.pt')
```

Also, if using YouTube streaming:

* Ensure `pafy` and `youtube-dl` are installed
* Add this to your imports:

```python
import pafy
pafy.set_backend("internal")
```

---

### ✅ **Step 6: Run the Streamlit App**

Navigate to the Streamlit app folder:

```bash
cd "streamlit-detection-tracking - app"
```

Run the app:

```bash
streamlit run app.py
```

---

### ✅ **Step 7: Use the Interface**

Once Streamlit opens in your browser:

* Upload an image or video
* Use a YouTube URL for live detection
* Or test with your webcam
* Optionally enable object tracking (ByteTrack or BoT-SORT)

---

### ✅ **Step 8: View Detection Results**

Detected objects will be highlighted in the output with bounding boxes and class labels. Classes are defined in your model's `data.yaml` or can be viewed with:

```python
model.names
```

---

### ✅ **Step 9: (Optional) Train Your Own YOLOv8 Model**

Use the Ultralytics CLI to train:

```bash
yolo task=detect mode=train model=yolov8n.pt data=your_data.yaml epochs=50 imgsz=640
```

Then replace the model path in `helper.py` with your new `.pt` file.

---

### ✅ **Step 10: Deploy or Push to Your GitHub** ( Not Needed )

Once everything works:

1. Add a `README.md`
2. Add `requirements.txt`
3. Push it to your GitHub repository

```bash
git init
git remote add origin https://github.com/yourusername/your-repo-name.git
git add .
git commit -m "Initial YOLOv8 waste detection app"
git push -u origin master
```

---

Let me know if you'd like a **guide for training your own dataset**, converting it to YOLO format, or deploying this online via Streamlit Cloud or Hugging Face Spaces!

## 🚀 Features

- 🧠 Powered by [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- 📹 Detect objects in uploaded images, videos, webcam, or YouTube streams
- ⚡ Real-time object tracking using ByteTrack and BoT-SORT
- 🌐 Easy-to-use Streamlit web interface
- 📦 Modular Python code structure for easy customization

---

## 🧱 Requirements

Create a virtual environment (optional but recommended) and install dependencies:

### 🔧 Python Version
- Python 3.8 or higher

### 📦 Dependencies

Install using pip:

```bash
pip install -r requirements.txt
