import streamlit as st
from ultralytics import YOLO
import numpy as np
from PIL import Image
import os

# Load model
MODEL_PATH = "models/best.pt"
model = YOLO(MODEL_PATH)

st.title("Helmet Detection App")
st.write("Upload an image, use webcam, or try a sample image.")

# Mode selector at top
mode = st.radio("Select Mode", ["Upload", "Webcam", "Sample"], horizontal=True)

def run_detection(pil_image):
    img_np = np.array(pil_image.convert("RGB"))
    img_bgr = img_np[:, :, ::-1]
    results = model.predict(img_bgr, conf=0.25, save=False)

    for r in results:
        if len(r.boxes) == 0:
            st.write("⚠️ No helmets detected!")
            return img_np
        else:
            img_with_boxes = r.plot()
            return img_with_boxes[:, :, ::-1]
    return img_np

if mode == "Upload":
    uploaded_file = st.file_uploader("Choose an image", type=["jpg","jpeg","png"])
    if uploaded_file:
        img = Image.open(uploaded_file)
        output_img = run_detection(img)
        st.image(output_img, caption="Detection Result")

elif mode == "Webcam":
    camera_image = st.camera_input("Take a photo")
    if camera_image:
        img = Image.open(camera_image)
        output_img = run_detection(img)
        st.image(output_img, caption="Detection Result")

elif mode == "Sample":
    sample_dir = "samples"
    sample_images = [f for f in os.listdir(sample_dir) if f.lower().endswith((".jpg",".png"))]
    choice = st.selectbox("Choose a sample image", sample_images)
    if choice:
        img_path = os.path.join(sample_dir, choice)
        img = Image.open(img_path)
        output_img = run_detection(img)
        st.image(output_img, caption=f"Sample: {choice}")
