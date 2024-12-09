# Image-Classification-using-ML-model
This project is a part of AICTE Internship on AI - TechSaksham -A CSR initiative of Microsoft &amp; SAP [ Project name: Implementation of ML Model for Image Classification]


# CIFAR-10 Image Classification Using Convolutional Neural Networks (CNN)

This project demonstrates the classification of images from the CIFAR-10 dataset using a Convolutional Neural Network (CNN). The CIFAR-10 dataset is a widely-used benchmark in computer vision, containing 60,000 32x32 color images across 10 classes.

---

## Project Overview

The goal of this project is to develop and train a CNN model that can classify images into one of the following 10 categories:

- Airplane
- Automobile
- Bird
- Cat
- Deer
- Dog
- Frog
- Horse
- Ship
- Truck

This implementation uses TensorFlow and Keras for model building, training, and evaluation.

---

## Dataset Details

- **Name**: CIFAR-10
- **Number of Images**: 60,000
  - Training: 50,000 images
  - Testing: 10,000 images
- **Image Size**: 32x32 pixels
- **Classes**: 10 (listed above)

---

## Key Features

- **Preprocessing**:
  - Normalized image pixel values to the range [0, 1].
  - Applied one-hot encoding to class labels.
- **Model Architecture**:
  - Convolutional Layers (Conv2D) for feature extraction.
  - MaxPooling Layers for down-sampling.
  - Dropout Layers for regularization.
  - Dense Layers for final classification.
- **Data Augmentation**:
  - Techniques such as flipping, rotation, and zoom to improve model generalization.
- **Optimization**:
  - Adam optimizer for adaptive learning rate adjustment.
  - Categorical Crossentropy loss function for multi-class classification.

---

   git clone <repository-link>
   cd <repository-folder>
