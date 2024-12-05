# FaceRecognitionModel
A model to predict the mood/emotion of a given picture from a face of a human beeing

Download the Colab notebook and open it in Google Colab.
The used FER Dataset is too huge for uploading. That is why it has to be downloaded from Kaggle: https://www.kaggle.com/datasets/nicolejyt/facialexpressionrecognition

The Tensorboard logs with the different Modeltraining runs are added in a ZIP File.
The best performing model is added as well.

# Face Recognition Emotion Detection Model - README

## Introduction

This notebook contains the code for developing a Convolutional Neural Network (CNN) model that predicts the emotional state of a person based on a given image of their face. The model has been trained to classify four emotions: **Angry**, **Happy**, **Sad**, and **Neutral**. The pre-trained model (`best_model.pth`) is used to make predictions on new input images.

## Prerequisites

- Python 3.7 or higher
- Jupyter Notebook environment (e.g., Google Colab, Jupyter Lab, or Jupyter Notebook)
- PyTorch and related libraries:
  - Install using:
    ```sh
    !pip install torch torchvision lightning pandas matplotlib tensorboard
    ```
- Google Drive mounted for storing datasets and model (optional if using Google Colab)

## Steps to Make Predictions with a Custom Input Face Image

### Step 1: Load the Pre-Trained Model

1. Ensure that the pre-trained model file (`best_model.pth`) is available in the specified directory (e.g., Google Drive or local directory).
2. Load the model as follows:
    ```python
    import torch
    model_path = '/content/drive/MyDrive/modelruns/saved_models/best_model.pth'  # Adjust the path as needed
    model = FaceRecModel()
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print('Model loaded successfully and ready for further use.')
    ```

### Step 2: Install Required Libraries

Install the `Pillow` library for image handling (if not already installed):
```sh
!pip install Pillow
```

### Step 3: Prepare the Custom Input Image

1. Select a facial image you want to classify.
2. Load and preprocess the image:
    ```python
    from PIL import Image
    from torchvision import transforms

    # Define the transformation steps
    transform = transforms.Compose([
        transforms.Resize((48, 48)),
        transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    # Load the image and apply the transformations
    image_path = '/content/drive/MyDrive/new_images/custom_image.jpg'  # Replace with your image path
    image = Image.open(image_path)
    image = transform(image)
    image = image.unsqueeze(0)  # Add batch dimension
    ```

### Step 4: Predict the Emotion

Run the following code to get a prediction:
```python
with torch.no_grad():
    output = model(image)
    _, predicted = torch.max(output, 1)
    emotion_mapping = {0: 'Angry', 1: 'Happy', 2: 'Sad', 3: 'Neutral'}
    predicted_emotion = emotion_mapping[predicted.item()]

print(f"Predicted Emotion: {predicted_emotion}")
```

### Step 5: Visualize the Input Image with Predicted Emotion

```python
import matplotlib.pyplot as plt

image_np = image.squeeze().numpy()
plt.imshow(image_np, cmap='gray')
plt.title(f"Predicted Emotion: {predicted_emotion}")
plt.axis('off')
plt.show()
```

## Summary

You have now successfully loaded a pre-trained model and used it to make a prediction on a custom input image. The model predicts one of four emotional states (Angry, Happy, Sad, or Neutral) based on the input facial image.

If you encounter any issues, ensure that all necessary packages are installed and that the model path is correct.



