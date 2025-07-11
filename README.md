This repository contains the code for a basic Human Activity Recognition (HAR) project using a Convolutional Neural Network (CNN).
The system takes in motion sensor data collected from smartphones and predicts which activity the person is performing (e.g., Walking, Sitting, etc.).

**Objective**
The goal is to build a simple CNN model that can learn patterns from accelerometer and gyroscope signals (x, y, z axes) and classify 6 different human activities using the UCI HAR Dataset.
**URL FOR DATASET:**
https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones 

Dataset
We used the UCI Human Activity Recognition Using Smartphones Dataset. Collected from 30 users performing 6 activities while carrying a smartphone
Sampled at 50Hz using:
Accelerometer (body and total) and Gyroscope

Activities Recognized:
Walking
Walking Upstairs
Walking Downstairs
Sitting
Standing
Laying

**Model Overview**
We used a 1D Convolutional Neural Network (CNN) to extract patterns from time-series signals.
The input shape is (128, 9) where:

128 is the number of time steps (2.56 seconds of data)

9 is the number of sensor features (x, y, z from 3 sensors)

Model Architecture:
Conv1D layer (64 filters) → learns local patterns

MaxPooling → reduces size, keeps important features

Conv1D layer (128 filters) → deeper patterns

Flatten → converts to 1D

Dense + Dropout → classification

Softmax output → one of 6 activity labels


**Evaluation**
The model was trained for 15 epochs
Used 80% for training, 20% for testing
Achieved a test accuracy of ~91.5%
Also plotted training/validation accuracy for detailed performance

**Tools Required**
Python 3.x
TensorFlow / Keras
NumPy
Matplotlib
Scikit-learn

