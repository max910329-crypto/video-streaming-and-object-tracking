# video-streaming-and-object-tracking
video streaming and object tracking HW demo

---


## hw1: Image Classification

* **Developed an image classification pipeline in PyTorch, including custom model design, training, and testing for a 100-class sports image dataset.**

### Introduction
1. For this assignment, you are required to build a neural network with PyTorch and train it 
to carry out a classification task.
2. To ensure that students can meet the assignment's requirements, the use of pretrained 
weights and existing models is not permitted

#### What to do?
1. You’re tasked with completing at least 3 Python Files: model.py, train.py, test.py
2. You need to implement your neural network inside model.py and named it as 
“ClassificationModel”. This network should be rendered accessible to train.py and 
test.py through import.
3. Within these files, you can import any package and design any additional classes or 
functions if you need. However, the utilization of ready-made neural network and 
pre-trained weight is forbidden. 
4. Discovery of any infringement of this cardinal rule will incur a penalty of a zero 
score for this assignment.

### Dataset
1. The dataset consists of sports images from 100 different categories. The size of each image 
is 224*224*3.
2. A total of 10000+ images are provided for training.
3. No validation set is given. Please split the training set yourself if you wish to evaluate your 
model’s performance.

---

## hw2: Object Tracking

* **Developed a multiple object tracking pipeline for video analysis, including person detection, bounding box association, unique ID assignment, trajectory visualization, and total people counting.**

### Objective
1. Implement visual multiple object tracking on videos
2. Using detection model + Hungarian algorithm
3. Using tracking model directly are NOT permitted
4. Calculate the total number of people appearing in the video
5. Output the tracking result video

### Steps
1. Choose a detection model and do detection for each frame.
  - You can use pre-trained weight or train by yourself.
  - It is only necessary to detect the ‘people’ category.
2. Using Hungarian algorithm to match the bounding boxes.
  - You can use existing sklearn function.
  - Decide on the cost factor by yourself. (IOU, distance, ReID similarity, etc.)
  - If same person leaves and re-enters the frame, they should be counted as a different individual.
3. Calculate the total number of people in the video.
4. Render tracking results and save the video

### Output Video Requirements

1. People count on the upper left corner
2. People bounding boxes
  - Box of different instance should be colored with 
difference colors.
  - Box of the same instance should maintain the same 
color within frames.
  - unique box_id labeled on box.
3. Trajectory visualization
  - Visualize each people’s trajectory


---

## hw3: Object Detection

* **Fine-tuned and customized an RT-DETR-based detection pipeline for car detection, including target-module modification and training on a custom GTA dataset.**

### Introduction
1. Train a neural network to do detection on our own dataset
2. Practice how to modify the target module in an existing model and implement it as a 
custom module.
3. Model : object detection algorithms
  - RT-DETR
    - You should use rdetr-resnet18 as your backbone model
    - if you choose another model as your backbone, you will get 0 points in HW3
  - The pretrained model weights are available

### Dataset
- GTA video dataset
- You only need to detect car 
  - Only one class
- 1596 training images, labels
- 227 validation images, labels
- 456 testing images


---

## hw4: Referring Expression Segmentation (RES)

* **Developed a vision-language segmentation pipeline by fine-tuning a CLIP-based model to generate pixel-level masks from natural language referring expressions.**

### Introduction
- In this assignment, you will fine-tune a language-guided image segmentation model 
on the given dataset to learn how natural-language inputs can condition pixel-level 
predictions.

### What to do?
- You’re tasked with completing at least 4 Python Files: dataset.py, model.py, train.py, 
test.py
- You need to implement your model inside model.py. This network should be rendered 
accessible to train.py and test.py through import.
- Within these files, you can import any package and design any additional classes or 
functions if you need. 
- Discovery of any infringement of this cardinal rule will incur a penalty of a zero 
score for this assignment.
- We have already implemented portions of the provided template. However, if you prefer, 
you may discard our implementation entirely.
- You can ONLY use openai/clip-vit-base-patch16 as your base model.

### Dataset
The dataset is built upon COCO images and contains a total of 17,596 samples for evaluation:
- Train: 13,811 samples
- Valid: 1,975 samples
- Test: 1,810 samples
