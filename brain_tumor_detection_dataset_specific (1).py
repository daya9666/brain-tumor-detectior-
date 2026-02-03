import cv2
import numpy as np
import os
from skimage.feature import hog
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, accuracy_score
import joblib
from joblib import Parallel, delayed
# 1. Load Dataset
def load_images_from_folder(folder):
   images = []
   labels = []
   label_map = {
       "glioma": 0,
       "meningioma": 1,
       "notumor": 2,
       "pituitary": 3
   }
   for label, label_num in label_map.items():
       path = os.path.join(folder, label)
       for file in os.listdir(path):
           img_path = os.path.join(path, file)
           try:
               img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
               if img is not None:
                   images.append(cv2.resize(img, (64, 64)))  # Resize to 64x64
                   labels.append(label_num)
           except Exception as e:
               print(f"Error reading {img_path}: {e}")
   return np.array(images), np.array(labels)
# 2. Extract HOG Features
def extract_hog_features_optimized(images):
   def process_image(img):
       feature, _ = hog(img, orientations=9, pixels_per_cell=(16, 16),
                        cells_per_block=(1, 1), block_norm='L2-Hys', visualize=True)
       return feature
   hog_features = Parallel(n_jobs=-1)(delayed(process_image)(img) for img in images)
   return np.array(hog_features)
# 3. Train the Model
def train_brain_tumor_model(dataset_path, model_save_path):
   print("Loading images...")
   images, labels = load_images_from_folder(dataset_path)
   # Check class distribution
   unique, counts = np.unique(labels, return_counts=True)
   print("Class Distribution:", dict(zip(unique, counts)))
   print(f"Loaded {len(images)} images.")
   print("Extracting HOG features...")
   hog_features = extract_hog_features_optimized(images)
   print(f"Extracted HOG features for {len(hog_features)} images.")
   print("Splitting dataset...")
   X_train, X_test, y_train, y_test = train_test_split(hog_features, labels, test_size=0.2, random_state=42)
   print(f"Training set size: {len(X_train)}, Test set size: {len(X_test)}")
   print("Training SVM model...")
   svm = SVC(kernel='linear', random_state=42)
   svm.fit(X_train, y_train)
   print("Evaluating model...")
   y_pred = svm.predict(X_test)
   accuracy = accuracy_score(y_test, y_pred)
   report = classification_report(y_test, y_pred, target_names=["Glioma", "Meningioma", "No Tumor", "Pituitary"])
   cm = confusion_matrix(y_test, y_pred)
   print(f"Model Accuracy: {accuracy * 100:.2f}%")
   print("Classification Report:")
   print(report)
   # Plot Confusion Matrix
   disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Glioma", "Meningioma", "No Tumor", "Pituitary"])
   disp.plot()
   # Save the model
   print(f"Saving model to {model_save_path}...")
   joblib.dump(svm, model_save_path)
   print("Model saved successfully.")
   return accuracy, report
# Example usage
if __name__ == "__main__":
   dataset_path = "brain/Training"  # Update this path to your dataset
   model_save_path = "brain_tumor_model_multi_class.pkl"
   train_brain_tumor_model(dataset_path, model_save_path)