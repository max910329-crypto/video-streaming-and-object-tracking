# video-streaming-and-object-tracking
video streaming and object tracking HW demo

---


## hw1: Image Classification

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

### Objective
#### Implement visual multiple object tracking on videos
#### Using detection model + Hungarian algorithm
##### Using tracking model directly are NOT permitted
##### Calculate the total number of people appearing in the video
#### Output the tracking result video

