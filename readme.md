# Rocket Objects Detection 🚀

## Table of Contents

1. [Project Goal](#project-goal)
2. [Problem Statement](#problem-statement)
3. [Impact of the Solution](#impact-of-the-solution)
4. [Data Collection](#data-collection)
5. [Data Exploration and Visualization (EDA)](#data-exploration-and-visualization-eda)
6. [Data Preprocessing](#data-preprocessing)
7. [Model Selection & Training](#model-selection--training)
8. [Model Evaluation](#model-evaluation)
9. [Deployment Readiness](#deployment-readiness)
10. [Conclusion](#conclusion)
11. [How to Run the Project](#how-to-run-the-project)

## Project Goal

The goal of this project is to develop a robust and efficient object detection model capable of accurately identifying and localizing various components of rockets in images and videos. This includes detecting engine flames, rocket bodies, and the tiny spec of the rocket once it is in space.

## Problem Statement

Develop an automated system to detect and classify rocket components (engine flames, rocket bodies, and space observation) in real-time during launches, reducing reliance on manual observation and improving accuracy and efficiency.

## Impact of the Solution

- **Enhanced Safety**: Real-time detection helps identify and address issues with the rockets promptly.
- **Efficiency**: Saves time and resources by automating the detection process.
- **Data Insights**: Enables better analysis and improvements in rocket design and performance.
- **Scalability**: Can monitor multiple launches simultaneously.
- **Innovation**: Advances computer vision and machine learning in aerospace applications.

## Data Collection

### Source of Data

The dataset is sourced from NASASpaceflight on Roboflow - Rocket Detect Computer Vision Project, which can be found at the following URL: [Rocket Detect](https://universe.roboflow.com/nasaspaceflight/rocket-detect).

### Composition of the Dataset

The dataset used for this project is specifically designed to train models for detecting various components of rockets. The dataset is composed of images labeled with three distinct classes:
1. **Engine Flames**: The fire produced by the rocket during launch.
2. **Rocket Body**: The main body of the launch vehicle.
3. **Space**: The tiny spec in the sky representing the rocket after it has ascended into space.

### Key Characteristics

- **Total Images**: The original dataset contains 12,303 images, but for this project, only half of them (approximately 6,150 images) were used and split into train, validation, and test folders.
- **Labeling**: Each image is accompanied by a label file that includes the class and bounding box coordinates for each detected object.
- **Negative Samples**: Some images have no labels (contain no detectable objects). These negative samples are crucial for training the model to distinguish between images with and without relevant objects, improving its overall accuracy, robustness, and reducing overfitting.

### Data Organization

The dataset is organized into images and corresponding labels, with each label file containing the class and bounding box coordinates for the objects in the image.

```
dataset/
├── images/
│   ├── train/
│   │   ├── image1.jpg
│   │   ├── image2.jpg
│   │   └── ...
│   ├── val/
│   │   ├── image1.jpg
│   │   ├── image2.jpg
│   │   └── ...
│   └── test/
│       ├── image1.jpg
│       ├── image2.jpg
│       └── ...
├── labels/
│   ├── train/
│   │   ├── image1.txt
│   │   ├── image2.txt
│   │   └── ...
│   ├── val/
│   │   ├── image1.txt
│   │   ├── image2.txt
│   │   └── ...
│   └── test/
│       ├── image1.txt
│       ├── image2.txt
│       └── ...
└── data.yaml
```

## Data Exploration and Visualization (EDA)

### Visualizing the Images & Labels

We visualized a random sample of images and their corresponding labels to understand the dataset better. The images were annotated with bounding boxes around the detected objects.

### Visualizing the Class Distribution

We plotted the class distribution to understand the balance of the dataset. This helps in identifying any class imbalance that might affect the model's performance.

## Data Preprocessing

### Splitting the Images into Train, Validation, and Test folders

The dataset was split into training, validation, and test sets to ensure that the model is trained and evaluated properly. The split was done in a 70-20-10 ratio.

## Model Selection & Training

The YOLOv8m model was selected for its balance of speed and accuracy, crucial for real-time rocket detection. Its pretrained weights accelerated training, and its robustness to variations ensured reliable detection. The user-friendly API and strong community support further enhanced its suitability for this project.

### Training Parameters

```python
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
```

## Model Evaluation

### Evaluation Metrics

- **Precision (P)**: The ratio of correctly predicted positive observations to the total predicted positives.
- **Recall (R)**: The ratio of correctly predicted positive observations to all observations in the actual class.
- **mAP50**: Mean Average Precision at IoU threshold of 0.50.
- **mAP50-95**: Mean Average Precision averaged over IoU thresholds from 0.50 to 0.95.

### Speed Metrics

The model's total processing time of **13.8ms** per image makes it suitable for real-time applications, capable of handling both 30 FPS and 60 FPS video streams without causing delays or frame drops.

### Confusion Matrix (Normalized)

The normalized confusion matrix reveals high true positive rates for 'Engine Flames' and 'Rocket Body', indicating strong detection performance for these classes. However, the 'Space' class had a lower true positive rate and higher misclassification rates.

### F1-Confidence Curve

The F1-confidence curve shows a peak F1 score of **0.81** at a confidence threshold of **0.312**.

### Visualizing Random Results

We visualized random results from the test set to qualitatively assess the model's performance.

## Deployment Readiness

The trained YOLOv8m model has been successfully tested on a real-world video of a ULA Atlas V launch, demonstrating the model's capability to accurately detect and annotate rocket objects in dynamic environments.

**Video Title**: ULA Atlas V launches final national security mission from Florida’s Space Coast

Channel: WKMG News 6 ClickOrlando

**Source**: [YouTube](https://www.youtube.com/watch?v=GYWr3FV9umU)

**Link of the processed video**: [YouTube](https://www.youtube.com/watch?v=iVRCdFLpDaE)

## Conclusion

Our project successfully developed a robust YOLOv8m-based model that accurately identifies and localizes rocket components such as engine flames, rocket bodies, and space observation in real-time. The model achieved an overall precision of **0.835** and recall of **0.792**, with an mAP50 of **0.826** and mAP50-95 of **0.46**, demonstrating its effectiveness in dynamic environments. The system's ability to process images at **13.8ms** per frame makes it suitable for real-time applications, enhancing efficiency in rocket launch monitoring.

**Further recommendations** include:
- Expanding the dataset to improve class balance.
- Training the model for longer than 100 epochs.
- Performing hyperparameter optimization.
- Integrating the model with real-time monitoring systems.
- Exploring advanced techniques like multi-scale training to enhance detection accuracy for smaller objects.

## How to Run the Project

1. **Clone the repository:**

```bash
git clone https://github.com/yourusername/rocket-objects-detection.git
cd rocket-objects-detection
```

2. **Install the required packages:**

```bash
pip install -r requirements.txt
```

3. **Download the dataset:**

Download the dataset from [Roboflow](https://universe.roboflow.com/nasaspaceflight/rocket-detect) and place it in the `dataset` directory.

4. **Run the training script:**

```bash
python train.py
```

5. **Evaluate the model:**

```bash
python evaluate.py
```

6. **Run inference on a video:**

```bash
python inference.py --video_path path_to_your_video.mp4
```

7. **View the results:**

The results will be saved in the `results` directory.

---
The model and Google Drive link is available upon request.
Feel free to contribute to this project by opening issues or submitting pull requests. 

Happy coding! 🚀
