import streamlit as st
import cv2
import numpy as np
from skimage.feature import hog
import joblib
# Load Pre-trained Model
@st.cache_resource
def load_model(model_path):
   return joblib.load(model_path)
# HOG Feature Extraction
def extract_hog_features(image):
   feature, _ = hog(image, orientations=9, pixels_per_cell=(16, 16),
                    cells_per_block=(1, 1), block_norm='L2-Hys', visualize=True)
   return feature.reshape(1, -1)
# Streamlit App
st.title("Brain Tumor Detection")
st.write("This application detects specific types of brain tumors (Glioma, Meningioma, Pituitary) or classifies as No Tumor using a pre-trained SVM model.")
# Upload Image
uploaded_file = st.file_uploader("Upload Brain MRI Image", type=["jpg", "png", "jpeg"])
if uploaded_file is not None:
   # Read and preprocess image
   file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
   image = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
   image_resized = cv2.resize(image, (64, 64))
   st.image(image, caption="Uploaded Image", use_container_width=True)
   # Load model
   model_path = "brain_tumor_model_multi_class.pkl"  # Update with the actual model file path
   model = load_model(model_path)
   # Extract HOG features and make prediction
   features = extract_hog_features(image_resized)
   prediction = model.predict(features)
   # Display prediction
   classes = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]
   result = classes[prediction[0]]
   st.write(f"Prediction: **{result}**")
