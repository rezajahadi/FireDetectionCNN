# Fire and Non-Fire Image Classifier

This repository contains a simple Convolutional Neural Network (CNN)-based classifier for detecting fire and non-fire images. The project includes the following components:

- **`main.ipynb`**: A script for training and defining the model.
- **`gui_fire_classifier.py`**: A GUI-based application for uploading and classifying images as "Fire" or "Non-Fire."

---

## Features

- A CNN model for binary classification (fire vs. non-fire).
- A user-friendly GUI for real-time image classification.
- Simple architecture designed for small datasets.

---

## Model Architecture

The CNN architecture includes:

1. **Convolutional Layers**: Two layers with batch normalization and max-pooling for feature extraction.
2. **Fully Connected Layers**: Three layers with batch normalization and dropout to reduce overfitting.
3. **Output Layer**: A final layer with a sigmoid activation function for binary classification.

---

## Challenges

This classifier faces the following challenges:

1. **Dataset Imbalance**: Unequal distribution of fire and non-fire samples may impact model performance.
2. **Small Dataset**: The model's generalizability is limited due to a lack of sufficient training data.
3. **Overfitting**: Dropout and batch normalization are used to mitigate overfitting, but the problem persists due to the small dataset size.

---

## Files

### `main.ipynb`

- Defines and trains the `FireDetectionCNN` model.
- Includes steps for data loading, preprocessing, and training.

### `gui_fire_classifier.py`

- Implements a graphical user interface (GUI) for real-time classification.
- Allows users to upload an image and view the prediction as "Fire" or "Non-Fire."

---

## Running the GUI
1. Ensure Python and the required libraries are installed. Install dependencies with:

**pip install -r requirements.txt**

2. Run the GUI script:

**python gui_fire_classifier.py**

3. Upload an image through the GUI to classify it as "Fire" or "Non-Fire."



