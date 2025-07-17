import streamlit as st
st.set_page_config(page_title="Waste Detector", layout="centered")

from ultralytics import YOLO
from PIL import Image
import torch
import torchvision.transforms as transforms
import torch.nn.functional as F
import os
import cv2
import numpy as np
import tempfile
from torchvision.models import mobilenet_v2
import pandas as pd

@st.cache_resource
def load_models():
    yolo_model = YOLO("Final_project/best.pt")
    mobilenet_model = mobilenet_v2(pretrained=False)
    classes = [
        'aerosol_cans', 'food_waste', 'general', 'glass_bottlles', 'glass_jars',
        'metal_cans', 'paper_cups', 'plastic_trash_bags', 'plastic_water_bottles',
        'plastic_cup_lids', 'magzines', 'newspaper'
    ]
    mobilenet_model.classifier[1] = torch.nn.Linear(mobilenet_model.last_channel, len(classes))
    mobilenet_model.load_state_dict(torch.load("Final_project/mobilenetv2_fine_classifier.pth", map_location='cpu'))
    mobilenet_model.eval()
    return yolo_model, mobilenet_model, classes

yolo_model, mobilenet_model, classes = load_models()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

coarse_to_fine = {
    "PLASTIC": ['plastic_trash_bags', 'plastic_water_bottles', 'plastic_cup_lids'],
    "METAL": ['aerosol_cans', 'metal_cans'],
    "PAPER": ['paper_cups', 'magzines', 'newspaper'],
    "GLASS": ['glass_bottlles', 'glass_jars'],
    "BIODEGRADABLE": ['food_waste'],
    "CARDBOARD": ['general'],
}

# Streamlit
st.sidebar.title(" Waste Detection App")
st.sidebar.markdown("Upload a waste image to detect and subclassify items.")
uploaded_file = st.sidebar.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

def classify_with_mobilenet_filtered(crop_img_path, yolo_label):
    image = Image.open(crop_img_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = mobilenet_model(input_tensor)
        probs = F.softmax(output, dim=1)[0].cpu()

    if yolo_label.upper() in coarse_to_fine:
        fine_classes = coarse_to_fine[yolo_label.upper()]
        fine_indices = [classes.index(cls) for cls in fine_classes]
        filtered_probs = probs[fine_indices]
        top_idx = torch.argmax(filtered_probs).item()
        predicted_class = fine_classes[top_idx]
        return predicted_class
    else:
        return "Unknown"


if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    results = yolo_model(tmp_path)
    image = Image.open(tmp_path).convert("RGB")
    img_np = np.array(image)
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    st.title(" Waste Detection and Subclassification")
    st.image(image, caption=" Uploaded Image", use_column_width=True)

    boxes = results[0].boxes
    annotated_img = img_bgr.copy()
    results_data = []
    crops = []

    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        class_name = yolo_model.names[int(box.cls[0])]
        crop = img_bgr[y1:y2, x1:x2]
        crop_path = f"crop_{i}_{class_name}.jpg"
        cv2.imwrite(crop_path, crop)

        subclass = classify_with_mobilenet_filtered(crop_path, class_name)
        crops.append((crop, subclass))

        label = f"{class_name} → {subclass}"
        cv2.rectangle(annotated_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(annotated_img, label, (x1, max(y1 - 10, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        results_data.append({
            "Object #": i + 1,
            "Main Category": class_name,
            "Subcategory": subclass
        })

    st.image(annotated_img, caption="Annotated Detection Image", channels="BGR", use_column_width=True)

    #Results table
    st.markdown(" Classification Table")
    df = pd.DataFrame(results_data)
    st.dataframe(df, use_container_width=True)

    #Show cropped objects
    st.markdown("Cropped Detected Objects")
    cols = st.columns(min(4, len(crops)))
    for idx, (crop, subclass) in enumerate(crops):
        with cols[idx % len(cols)]:
            st.image(crop, caption=subclass, use_column_width=True)
