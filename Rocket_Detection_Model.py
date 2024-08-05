#!/usr/bin/env python
# coding: utf-8

# # Rocket Objects Detection 🚀

# # 1. Problem Definition

# ## 1.1. Project Goal
# 

# ## 1.2. Problem Statement
# 

# ## 1.3. Impact of the Solution
# 

# ---

# # 2. Data Collection

# ## 2.1. Source of Data
# The dataset is sourced from NASASpaceflight on Roboflow - Rocket Detect Computer Vision Project, which can be found at the following URL: https://universe.roboflow.com/nasaspaceflight/rocket-detect
# 

# ## 2.2. Composition of the Dataset
# 

# ## 2.3. Data Organization
# 
# 

# ---

# # 3. Data Exploration and Visualization (EDA)

# In[ ]:


import os
import random

import cv2
import torch
import shutil

import matplotlib.pyplot as plt
from ultralytics import YOLO
from google.colab import drive

print(torch.cuda.is_available())
print(torch.cuda.device_count())


drive.mount('/content/drive', force_remount=True)
dataset_path = '/content/drive/My Drive/yolo_dataset/data.yaml'
model_path = '/content/drive/My Drive/yolo_dataset/last_v8m.pt'
save_dir = '/content/drive/My Drive/yolo_dataset'


# ## 3.1. Visualizing the Images & Labels

# In[ ]:


images_folder = '6k_dataset/images'
labels_folder = '6k_dataset/labels'

labels = {
    0: 'Engine Flames',
    1: 'Rocket Body',
    2: 'Space'
}

# Load a random sample of images and their labels
def load_random_samples(num_samples=5):
    image_files = [f for f in os.listdir(images_folder) if f.endswith('.jpg')]
    random_samples = random.sample(image_files, num_samples)
    
    samples = []
    for image_file in random_samples:
        image_path = os.path.join(images_folder, image_file)
        label_file = image_file.replace('.jpg', '.txt')
        label_path = os.path.join(labels_folder, label_file)
        
        with open(label_path, 'r') as f:
            label_data = f.readlines()
        
        samples.append((image_path, label_data))
    
    return samples

# Visualize images with bounding boxes
def visualize_samples(samples):
    for image_path, label_data in samples:
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        for line in label_data:
            parts = line.strip().split()
            label = int(parts[0])
            x_center = float(parts[1])
            y_center = float(parts[2])
            width = float(parts[3])
            height = float(parts[4])
            
            img_height, img_width, _ = image.shape
            x_center *= img_width
            y_center *= img_height
            width *= img_width
            height *= img_height
            
            x_min = int(x_center - width / 2)
            y_min = int(y_center - height / 2)
            x_max = int(x_center + width / 2)
            y_max = int(y_center + height / 2)
            
            cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)
            cv2.putText(image, labels[label], (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
        
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        plt.axis('off')
        plt.show()

samples = load_random_samples(num_samples=5)
visualize_samples(samples)


# ![sample1](https://i.imgur.com/eqvFsFc.png)
# ![sample2](https://i.imgur.com/U7ljJLI.png)
# ![sample3](https://i.imgur.com/Xz661o0.png)
# ![sample4](https://i.imgur.com/cDCH8tX.png)

# ## 3.2. Visualizing the Class Distribution

# In[ ]:


# Count the class distribution
def count_class_distribution():
    class_counts = {0: 0, 1: 0, 2: 0}
    
    label_files = [f for f in os.listdir(labels_folder) if f.endswith('.txt')]
    
    for label_file in label_files:
        label_path = os.path.join(labels_folder, label_file)
        
        with open(label_path, 'r') as f:
            label_data = f.readlines()
        
        for line in label_data:
            parts = line.strip().split()
            label = int(parts[0])
            class_counts[label] += 1
    
    return class_counts

def plot_class_distribution(class_counts):
    classes = [labels[key] for key in class_counts.keys()]
    counts = [class_counts[key] for key in class_counts.keys()]
    
    plt.figure(figsize=(10, 6))
    plt.bar(classes, counts, color=['blue', 'green', 'red'])
    plt.xlabel('Class')
    plt.ylabel('Count')
    plt.title('Class Distribution in Dataset')
    plt.show()

# Count class distribution and plot it
class_counts = count_class_distribution()
plot_class_distribution(class_counts)


# ![classdist](https://i.imgur.com/ClHb4qx.png)

# # 4. Data Preprocessing

# ## 4.1. Splitting the Images into Train, Validation and Test folders

# In[ ]:


dataset_path = '6k_dataset'
images_path = os.path.join(dataset_path, 'images')
labels_path = os.path.join(dataset_path, 'labels')

# Get all image files
image_files = [f for f in os.listdir(images_path) if f.endswith('.jpg')]

# Shuffle the files
random.shuffle(image_files)

# Calculate split indices
total_images = len(image_files)
train_split = int(total_images * 0.7)
val_split = int(total_images * 0.2)

# Split the files
train_files = image_files[:train_split]
val_files = image_files[train_split:train_split + val_split]
test_files = image_files[train_split + val_split:]

# Copy files to their directories
def copy_files(file_list, dest_images_path, dest_labels_path):
    os.makedirs(dest_images_path, exist_ok=True)
    os.makedirs(dest_labels_path, exist_ok=True)
    for file in file_list:
        shutil.copy(os.path.join(images_path, file), dest_images_path)
        label_file = file.replace('.jpg', '.txt')
        if os.path.exists(os.path.join(labels_path, label_file)):
            shutil.copy(os.path.join(labels_path, label_file), dest_labels_path)

# Create directories and copy files
copy_files(train_files, 'yolo_dataset/train/images', 'yolo_dataset/train/labels')
copy_files(val_files, 'yolo_dataset/val/images', 'yolo_dataset/val/labels')
copy_files(test_files, 'yolo_dataset/test/images', 'yolo_dataset/test/labels')


# # 5. Model Selection & Training

# The YOLOv8m model was selected for its balance of speed and accuracy, crucial for real-time rocket detection. Its pretrained weights accelerate  training, and its robustness to variations ensures reliable detection. The user-friendly API and strong community support further enhance its suitability for this project. Other models were tested first, namely yolov8s and yolov8s pretrained, but yolov8m pretrained yieleded the best results.

# In[ ]:


model = YOLO('yolov8m.pt')

training_params = {
    'data': dataset_path,  
    'imgsz': 640, 
    'epochs': 100, 
    'augment': True,  
    'patience': 10,  # Early Stopping
    'save_period': 1,  
    'save': True,  
    'resume': True,  
    'project': save_dir,
    'name': 'my_experiment'
}

model.train(**training_params)


# # 6. Model Evaluation

# In[ ]:


model = YOLO(model_path)

# Evaluate the model on the test set
results = model.val(data=dataset_path, save_dir=save_dir)

print(results)


# ## 6.1. Evaluation Metrics:
# 

# ```plaintext
#                  Class     Images  Instances      Precision  Recall     mAP50       mAP50-95)
#                    all       1221       1390      0.835      0.792      0.826       0.46
#          Engine Flames        562        570      0.902      0.911      0.945      0.603
#            Rocket Body        622        633      0.914      0.889      0.925       0.55
#                  Space        165        187      0.689      0.578      0.608      0.228
# ```

# - **Precision (P)**: The ratio of correctly predicted positive observations to the total predicted positives. High precision indicates a low false positive rate.
# - **Recall (R)**: The ratio of correctly predicted positive observations to all observations in the actual class. High recall indicates a low false negative rate.
# - **mAP50**: Mean Average Precision at IoU threshold of 0.50. It is the average precision across all classes.
# - **mAP50-95**: Mean Average Precision averaged over IoU thresholds from 0.50 to 0.95. It provides a more comprehensive evaluation of the model's performance.
# 
# ### Analysis
# - **Overall Performance**: The model achieved an overall precision of **0.835** and recall of **0.792**, with an mAP50 of **0.826** and mAP50-95 of **0.46**. This indicates good overall performance, with a balanced ability to correctly identify and localize objects.

# ## 6.2. Speed Metrics:

# ```plaintext
# Speed: 0.2ms preprocess, 11.4ms inference, 0.0ms loss, 2.2ms postprocess per image
# Total inference time per image: 13.8ms
# ```

# This speed is suitable for real-time applications.
# 
# For a video running at 30 frames per second (FPS), each frame needs to be processed within approximately 33.3ms (1000ms/30). For a 60 FPS video, each frame needs to be processed within approximately 16.7ms (1000ms/60). The model's total processing time of **13.8ms** per image comfortably fits within these time constraints, making it capable of handling both 30 FPS and 60 FPS video streams without causing delays or frame drops.

# ## 6.3. Confusion Matrix (Normalized)

# ![confusionmatrix](https://i.imgur.com/BiFnMuW.png)

# The normalized confusion matrix reveals high true positive rates for 'Engine Flames' **(0.92)** and 'Rocket Body' **(0.91)**, indicating strong detection performance for these classes. However, the 'Space' class had a lower true positive rate **(0.64)** and higher misclassification rates. This suggests difficulty in distinguishing the tiny spec of the rocket in space from other classes, likely due to its small size and less distinct features, as well as the class imbalance present in the dataset.

# ## 6.4. F1-Confidence Curve

# ![f1confidence](https://i.imgur.com/3wyVRR7.png)

# The F1-confidence curve shows a peak F1 score of **0.81** at a confidence threshold of **0.312**. This bell-shaped curve indicates that at this confidence level, the model achieves the best balance between precision and recall across all classes.

# ## 6.5. Visualizing Random Results

# In[ ]:


for result in results:
    annotated_image = result.plot()  # Get the annotated image
    plt.imshow(annotated_image)
    plt.axis('off') 
    plt.show()


# ![results](https://i.imgur.com/0KZGDvh.png)
# ![results](https://i.imgur.com/vybRNDm.png)
# ![results](https://i.imgur.com/uQvfFRw.png)
# ![results](https://i.imgur.com/JLeVxiY.png)
# ![results](https://i.imgur.com/IGEDHuU.png)
# ![results](https://i.imgur.com/hTYIWZT.png)

# # 7. Deployment Readiness

# ULA Atlas V launches final national security mission from Florida’s Space Coast
# 
# WKMG News 6 ClickOrlando
# 
# Source: https://www.youtube.com/watch?v=GYWr3FV9umU
# Link of the processed video: https://www.youtube.com/watch?v=iVRCdFLpDaE

# In[ ]:


model = YOLO(model_path)

video_path = '/content/drive/My Drive/yolo_dataset/launch_video.mp4'

# Directory to save the annotated video
experiment_name = 'video_inference'

# Run inference on the video with stream=True (to avoid memory crashes)
results = model.predict(source=video_path, stream=True, save=True, project=save_dir, name=experiment_name)

# Process the results
for r in results:
    boxes = r.boxes  # Boxes object for bbox outputs
    masks = r.masks  # Masks object for segment masks outputs
    probs = r.probs  # Class probabilities for classification outputs

print(f"Annotated video saved to {save_dir}/{experiment_name}")


# # 8. Conclusion

# In[ ]:




