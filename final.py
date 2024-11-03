import torch
import timm
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from PIL import Image
import pandas as pd
import numpy as np
import os

# Define class labels
class_columns = ['Angioectasia', 'Bleeding', 'Erosion', 'Erythema', 'Foreign Body', 
                 'Lymphangiectasia', 'Normal', 'Polyp', 'Ulcer', 'Worms']

# Define transformations
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

test_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Define dataset paths
train_data_dir = '/kaggle/input/capsule-vision-2024-challenge/Dataset/training'
val_data_dir = '/kaggle/input/capsule-vision-2024-challenge/Dataset/validation'
test_data_dir = '/kaggle/input/testing/Testing set/Images'

# Load datasets
train_dataset = datasets.ImageFolder(root=train_data_dir, transform=train_transforms)
val_dataset = datasets.ImageFolder(root=val_data_dir, transform=test_transforms)

# Data loaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Define the Swin Transformer model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
swin_model = timm.create_model('swin_tiny_patch4_window7_224', pretrained=True, num_classes=len(class_columns))
swin_model.to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = Adam(swin_model.parameters(), lr=0.0001)

# Training function
def train(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        correct += (outputs.argmax(1) == labels).sum().item()
        total += labels.size(0)
    return total_loss / len(train_loader), 100 * correct / total

# Validation function
def validate(model, val_loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)
    return total_loss / len(val_loader), 100 * correct / total

# Train and validate the model
epochs = 10
best_val_accuracy = 0

for epoch in range(epochs):
    train_loss, train_accuracy = train(swin_model, train_loader, criterion, optimizer, device)
    val_loss, val_accuracy = validate(swin_model, val_loader, criterion, device)
    
    print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.2f}% - Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.2f}%")
    
    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        torch.save(swin_model.state_dict(), 'best_swin_model.pth')
        print("Model saved!")

# Load the best model for testing
swin_model.load_state_dict(torch.load('best_swin_model.pth'))
swin_model.eval()

# Custom Dataset for Test Data
class TestDataset(torch.utils.data.Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.image_paths = [os.path.join(image_dir, img) for img in os.listdir(image_dir) if img.endswith(('.png', '.jpg', '.jpeg'))]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, img_path

# Test DataLoader
test_dataset = TestDataset(test_data_dir, transform=test_transforms)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Function to save predictions to CSV for any dataset
def save_predictions_to_csv(model, dataloader, output_csv, device):
    model.eval()
    predictions = []
    with torch.no_grad():
        for images, image_paths in dataloader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.nn.functional.softmax(outputs, dim=1).cpu().numpy()
            predicted_classes = np.argmax(probs, axis=1)
            
            for i, img_path in enumerate(image_paths):
                pred_row = [img_path] + list(probs[i]) + [class_columns[predicted_classes[i]]]
                predictions.append(pred_row)

    # Create DataFrame and save to CSV
    columns = ['image_path'] + class_columns + ['predicted_class']
    df = pd.DataFrame(predictions, columns=columns)
    df.to_csv(output_csv, index=False)
    print(f"Predictions saved to {output_csv}")

# Save predictions for training and validation datasets
train_test_dataset = TestDataset(train_data_dir, transform=train_transforms)
train_test_loader = DataLoader(train_test_dataset, batch_size=32, shuffle=False)
save_predictions_to_csv(swin_model, train_test_loader, 'train_predictions.csv', device)

val_test_dataset = TestDataset(val_data_dir, transform=test_transforms)
val_test_loader = DataLoader(val_test_dataset, batch_size=32, shuffle=False)
save_predictions_to_csv(swin_model, val_test_loader, 'val_predictions.csv', device)

# Run the prediction function for the test dataset
save_predictions_to_csv(swin_model, test_loader, 'test_predictions.csv', device)
