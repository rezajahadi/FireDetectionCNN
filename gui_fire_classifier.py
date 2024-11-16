import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import torch
from torchvision import transforms
import numpy as np


class FireDetectionCNN(nn.Module):
    def __init__(self):
        super(FireDetectionCNN, self).__init__()

        # Convolutional Layers with Batch Normalization
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        # Fully Connected Layers with Dropout and Batch Normalization
        self.fc1 = nn.Linear(56 * 56 * 64, 128)  # Adjusted based on input size and pooling
        self.dropout1 = nn.Dropout(0.5)
        self.bn_fc1 = nn.BatchNorm1d(128)

        self.fc2 = nn.Linear(128, 64)
        self.dropout2 = nn.Dropout(0.5)
        self.bn_fc2 = nn.BatchNorm1d(64)

        self.fc3 = nn.Linear(64, 32)
        self.dropout3 = nn.Dropout(0.5)

        # Output Layer
        self.fc4 = nn.Linear(32, 1)  # Binary classification, so output is 1
        self.sigmoid = nn.Sigmoid()  # Use sigmoid activation for binary output

    def forward(self, x):
        # First Conv Layer
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)

        # Second Conv Layer
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)

        # Flatten the output from conv layers
        x = x.view(x.size(0), -1)

        # Fully Connected Layers
        x = F.relu(self.bn_fc1(self.fc1(x)))
        x = self.dropout1(x)

        x = F.relu(self.bn_fc2(self.fc2(x)))
        x = self.dropout2(x)

        x = F.relu(self.fc3(x))
        x = self.dropout3(x)

        # Output Layer with Sigmoid Activation
        x = self.sigmoid(self.fc4(x))
        return x



# Load the trained model
class FireDetectionApp:
    def __init__(self, master, model, device):
        self.master = master
        self.master.title("Fire Detection Classifier")
        self.master.geometry("600x400")
        
        # Model and device setup
        self.model = model
        self.device = device
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

        # UI Elements
        self.label = tk.Label(master, text="Upload an Image to Classify", font=("Arial", 16))
        self.label.pack(pady=20)

        self.image_label = tk.Label(master, text="No image uploaded", font=("Arial", 12))
        self.image_label.pack(pady=10)

        self.result_label = tk.Label(master, text="", font=("Arial", 14))
        self.result_label.pack(pady=20)

        self.upload_button = tk.Button(master, text="Upload Image", command=self.upload_image, font=("Arial", 14))
        self.upload_button.pack(pady=10)

    def upload_image(self):
        # File dialog for selecting an image
        file_path = filedialog.askopenfilename(
            title="Select an Image File",
            filetypes=[
                ("All Files", "*.*"),
            ]
        )
        if not file_path or not os.path.exists(file_path):
            print("No file selected or file does not exist.")
            return
        
        # Load and display the image
        img = Image.open(file_path)
        img_resized = img.resize((224, 224))  # Resize for display
        img_tk = ImageTk.PhotoImage(img_resized)
        self.image_label.configure(image=img_tk, text="", compound=tk.TOP)
        self.image_label.image = img_tk

        # Predict the class
        self.predict_image(file_path)


    def predict_image(self, file_path):
        # Load the image and preprocess
        img = Image.open(file_path).convert("RGB")
        img_tensor = self.transform(img).unsqueeze(0).to(self.device)

        # Perform inference
        self.model.eval()
        with torch.no_grad():
            print(f"the shape of image is {img_tensor.shape}")
            output = self.model(img_tensor).view(-1)
            prediction = (output > 0.5).int()
            print(f"the value of output is {output}")
        
        # Display the result
        class_names = ["Fire", "Non-Fire"]
        result_text = f"Prediction: {class_names[prediction]}"
        self.result_label.config(text=result_text, fg="green" if prediction == 1 else "red")

# Load your trained model
model = FireDetectionCNN()
model.load_state_dict(torch.load("FireDetector.pth", map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu")))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Create GUI
root = tk.Tk()
app = FireDetectionApp(root, model, device)
root.mainloop()
