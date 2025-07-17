
# ♻️ Waste Classification using MobileNetV2

This project implements an **image classification model using MobileNetV2** to categorize waste into multiple fine-grained classes (e.g., plastic, paper, metal, food waste, etc.). It forms the second stage of our hybrid AI-based waste classification pipeline, enabling better decision-making in automated waste sorting systems.

---

## 🛠️ Step-by-Step Guide: Training MobileNetV2 Waste Classifier in Google Colab

### ✅ Step 1: Prepare Your Dataset

Ensure your dataset is organized using the **ImageFolder format**, with each waste category placed in its respective subfolder:

```
dataset/
  ├── Plastic/
  ├── Paper/
  ├── Metal/
  ├── Food_Waste/
  ├── ...
```

Upload the dataset to **Google Drive**, e.g., inside:

```
/MyDrive/Wasteclass_mobilenetv2/dataset/
```

---

### ✅ Step 2: Open Google Colab

Create or open a new Google Colab notebook and make sure GPU is enabled:
`Runtime > Change runtime type > Hardware accelerator > GPU`

---

### ✅ Step 3: Mount Google Drive

Mount your Google Drive to access the dataset and save model checkpoints:

```python
from google.colab import drive
drive.mount('/content/drive')
```

---

### ✅ Step 4: Install Required Libraries

Colab already includes most libraries, but run the following to ensure compatibility:

```python
!pip install torch torchvision matplotlib
```

---

### ✅ Step 5: Load and Preprocess Dataset

Use torchvision’s transforms for resizing and normalization:

```python
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])
```

Create DataLoaders for training and validation:

```python
train_dataset = datasets.ImageFolder(train_dir, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

val_dataset = datasets.ImageFolder(val_dir, transform=transform)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
```

---

### ✅ Step 6: Modify and Train MobileNetV2

Fine-tune MobileNetV2 for your dataset:

```python
model = mobilenet_v2(pretrained=False)
model.classifier[1] = nn.Linear(model.last_channel, 12)  # Change 12 if using different number of classes
```

Set up training loop using CrossEntropyLoss and Adam optimizer:

```python
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
```

Train for desired epochs and monitor accuracy:

```python
for epoch in range(num_epochs):
    model.train()
    ...
    optimizer.step()
```

---

### ✅ Step 7: Save the Trained Model

After training, save your model checkpoint to Google Drive:

```python
torch.save(model.state_dict(), '/content/drive/MyDrive/Wasteclass_mobilenetv2/mobilenetv2_final.pth')
```

---

### ✅ Step 8: Load Model for Inference (Optional)

To reuse the trained model later:

```python
model.load_state_dict(torch.load('/content/drive/MyDrive/Wasteclass_mobilenetv2/mobilenetv2_final.pth'))
model.eval()
```

---

## 🚀 Features

* ✅ Lightweight, fast, and efficient classification using **MobileNetV2**
* ✅ Custom-trained on 12 waste categories for fine-grained classification
* ✅ Designed to work seamlessly with YOLOv8 object detection output
* ✅ Easy training and reproducibility using Google Colab + Google Drive

---

## 📦 Requirements (For Local Use)

For replicating or running the training script locally:

```txt
torch>=1.13.1
torchvision>=0.14.1
matplotlib
```

Install using:

```bash
pip install -r requirements.txt
```

---

## 🔗 Dataset

A custom dataset of over **30,000 waste images** organized in 12 classes was used for training. If you wish to try a public alternative, consider:
[Roboflow Garbage Classification Dataset](https://universe.roboflow.com/project-1ycjd/garbage-classification-jxps3/dataset)

---

Let me know if you’d like this added to a `.md` file directly or if you want to add images, plots, or class-wise results too!
