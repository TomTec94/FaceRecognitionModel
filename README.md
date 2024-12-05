# FaceRecognitionModel
A model to predict the mood/emotion of a given picture from a face of a human beeing

Download the Colab notebook and open it in Google Colab.
The used FER Dataset is too huge for uploading. That is why it has to be downloaded from Kaggle: https://www.kaggle.com/datasets/nicolejyt/facialexpressionrecognition

The Tensorboard logs with the different Modeltraining runs are added in a ZIP File.
The best performing model is added as well.


# Face Recognition Model for Emotion Detection

This project aims to implement a tool that detects the emotional state of a person based on facial expressions. The system uses a deep learning model built with PyTorch to classify images into different emotional states.

## Table of Contents

- [Installation](#installation)
- [Making Predictions](#making-predictions)
- [Usage](#usage)

## Installation

### Step 1: Set up the Environment

First, you need to connect Google Drive to access the dataset and install the required packages. Use Google Colab for seamless access to Google Drive.

To prepare the environment, run the following code:

```python
from google.colab import drive
drive.mount('/content/drive')
!pip install torch torchvision pandas matplotlib torchviz tensorboard lightning
```

## Making Predictions

### Step 2: Load the Model for Prediction

Since the model has already been trained, you can load it directly to make predictions on new images. Use the following code to load the model:

```python
model = setup_model_for_prediction('/content/drive/MyDrive/modelruns/saved_models/best_model.pth')
```

### Step 3: Predict Emotion from New Images

To make predictions on new images:

```python
predict_emotion(model, '/content/drive/MyDrive/new_images/[image_name].jpg')
```

This function will output the predicted emotion for the given image.

## Usage

1. **Environment Setup**: Use the provided steps to set up the environment for running the model.
2. **Load the Model**: Use `setup_model_for_prediction()` to load the pre-trained model.
3. **Make Predictions**: Use `predict_emotion()` to make predictions on unseen images.

With these steps, the model is ready to use for recognizing emotional states from facial expressions in images.

