DATESET LINK : https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset


Brain Tumor Detection Using Machine Learning

Project Overview

This project involves building a machine learning model to detect and classify brain tumors using MRI images. The model identifies the type of tumor (Glioma, Meningioma, Pituitary) or determines if no tumor is present. A Streamlit-based web application is also included to allow users to upload MRI images and get predictions.

Folder Structure

brain-tumor-detection/
├── brain/
│   ├── Training/
│   │   ├── glioma/         # MRI images with glioma tumors
│   │   ├── meningioma/     # MRI images with meningioma tumors
│   │   ├── notumor/        # MRI images with no tumors
│   │   └── pituitary/      # MRI images with pituitary tumors
│   ├── Testing/ (optional) # Additional images for testing
├── brain_tumor_detection_updated.py    # Training script
├── brain_tumor_detection_streamlit.py  # Streamlit web application
├── brain_tumor_model_multi_class.pkl   # Trained model file
└── README.md                           # Documentation file

How It Works

1. Dataset Preparation
• Organize MRI images into the following classes:
•
Glioma
•
Meningioma
•
No Tumor
•
Pituitary
• Place the images in the brain/Training/ directory in their respective folders.

2. Model Training
• The training script (brain_tumor_detection_updated.py):
• Loads the dataset from the Training folder.
• Extracts features using
Histogram of Oriented Gradients (HOG)
.
• Trains a
Support Vector Machine (SVM)
for multi-class classification.
• Saves the trained model as brain_tumor_model_multi_class.pkl.

3. Streamlit Application
• The Streamlit app (brain_tumor_detection_streamlit.py):
• Allows users to upload an MRI image.
• Preprocesses the image and extracts HOG features.
• Loads the trained model to make predictions.
• Displays the predicted tumor type or confirms “No Tumor.”

Setup Instructions

1. Clone the Repository

git clone
https://github.com/your-repo/brain-tumor-detection.git
cd brain-tumor-detection

2. Install Dependencies

pip install -r requirements.txt
Include libraries like:
• streamlit
• scikit-learn
• opencv-python
• numpy
• scikit-image
• joblib

3. Train the Model
• Run the training script:
python brain_tumor_detection_updated.py
• The model will be saved as brain_tumor_model_multi_class.pkl.

4. Run the Streamlit App
• Launch the app to predict tumor types:
streamlit run brain_tumor_detection_streamlit.py

Usage
1. Open the Streamlit app in your browser.
2. Upload an MRI image (formats: .jpg, .png, .jpeg).
3. The app will display:
• The uploaded image.
• The predicted tumor type (Glioma, Meningioma, Pituitary) or No Tumor.

Results
• The model achieves an accuracy of ~72% on the current dataset (can vary).
• Includes detailed evaluation using a
confusion matrix
and
classification report
.

Future Improvements
• Use
Convolutional Neural Networks (CNNs)
for better accuracy.
• Add more diverse datasets to improve generalization.
• Implement real-time MRI image analysis.

Contributors
•
SRAV TECH

Let me know if you want to adjust anything for your project!
