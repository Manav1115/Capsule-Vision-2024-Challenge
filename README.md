![ChallengeHeader](https://github.com/user-attachments/assets/e75f510b-02a8-4fec-b133-11f4ab6c828d)
# Capsule Vision Challenge 2024: Multi-Class Abnormality Classification for Video Capsule Endoscopy

## Challenge Overview
The aim of the challenge is to provide an opportunity
for the development, testing and evaluation of AI models
for automatic classification of abnormalities captured in
VCE video frames. It promotes the development of vendor-independent and
generalized AI-based models for automatic abnormality
classification pipeline with 10 class labels namely angioectasia, bleeding, erosion, erythema, foreign body,
lymphangiectasia, polyp, ulcer, worms, and normal.

## Dataset 
The training and validation dataset has been developed using
three publicly available (SEE-AI project dataset, KID,
and Kvasir-Capsule dataset) and one private dataset (AIIMS) VCE datasets. The training and validation dataset
consist of 37,607 and 16,132 VCE frames respectively mapped to 10 class labels namely angioectasia, bleeding, erosion, erythema, foreign body, lymphangiectasia, polyp, ulcer, worms,
and normal.
| Type of Data | Source Dataset | Angioectasia | Bleeding | Erosion | Erythema | Foreign Body | Lymphangiectasia | Normal | Polyp | Ulcer | Worms |
|--------------|----------------|--------------|----------|---------|----------|---------------|------------------|--------|-------|-------|-------|
| Training     | KID            | 18           | 3        | 0       | 0        | 0             | 6                | 315    | 34    | 0     | 0     |
|              | KVASIR         | 606          | 312      | 354     | 111      | 543           | 414              | 24036  | 38    | 597   | 0     |
|              | SEE-AI         | 530          | 519      | 2340    | 580      | 249           | 376              | 4312   | 1090  | 0     | 0     |
|              | AIIMS          | 0            | 0        | 0       | 0        | 0             | 0                | 0      | 0     | 66    | 158   |
| **Total Frames** |                | **1154**     | **834**  | **2694**| **691**  | **792**       | **796**          | **28663**| **1162**| **663**| **158** |
| Validation   | KID            | 9            | 2        | 0       | 0        | 0             | 3                | 136    | 15    | 0     | 0     |
|              | KVASIR         | 260          | 134      | 152     | 48       | 233           | 178              | 10302  | 17    | 257   | 0     |
|              | SEE-AI         | 228          | 223      | 1003    | 249      | 107           | 162              | 1849   | 468   | 0     | 0     |
|              | AIIMS          | 0            | 0        | 0       | 0        | 0             | 0                | 0      | 0     | 29    | 68    |
| **Total Frames** |                | **497**      | **359**  | **1155**| **297**  | **340**       | **343**          | **12287**| **500** | **286** | **68** |

### Dataset Structure
The images are organized into their respective classes for both the training and validation datasets as shown below:
```bash
Dataset/
├── training
│   ├── Angioectasia
│   ├── Bleeding
│   ├── Erosion
│   ├── Erythema
│   ├── Foreign Body
│   ├── Lymphangiectasia
│   ├── Normal
│   ├── Polyp
│   ├── Ulcer
│   └── Worms
│   └── training_data.xlsx
└── validation
    ├── Angioectasia
    ├── Bleeding
    ├── Erosion
    ├── Erythema
    ├── Foreign Body
    ├── Lymphangiectasia
    ├── Normal
    ├── Polyp
    ├── Ulcer
    └── Worms
    └── validation_data.xlsx
```

This project uses a Swin Transformer model for multi-class classification of medical images into categories such as Angioectasia, Bleeding, Erosion, and more. The model is trained, validated, and tested on images, and predictions are saved to CSV files for further analysis.

### Project Structure
train_data_dir: Directory containing training images.
val_data_dir: Directory containing validation images.
test_data_dir: Directory containing test images for final evaluation.
swin_model.pth: Saved model weights with the best validation accuracy.
train_predictions.csv: CSV file with predictions on training data.
val_predictions.csv: CSV file with predictions on validation data.
test_predictions.csv: CSV file with predictions on test data.

### Requirements
To run the project, you need to install the following Python libraries:

torch
timm
torchvision
Pillow
pandas
numpy
You can install these requirements using:
```bash
pip install torch torchvision timm Pillow pandas numpy
```

### Dataset
Place the images into the following directories:

train_data_dir: Directory with training images.
val_data_dir: Directory with validation images.
test_data_dir: Directory with test images.
The class labels are:
```bash
['Angioectasia', 'Bleeding', 'Erosion', 'Erythema', 'Foreign Body',  'Lymphangiectasia', 'Normal', 'Polyp', 'Ulcer', 'Worms']
```

### Training and Validation
Data Preparation:

Images are resized to (224, 224), normalized, and augmented with random horizontal flips for training.
Validation data is resized and normalized without augmentation.
Model Architecture:

Uses Swin Transformer (from timm library) with num_classes set to the number of classes.
Training:

The model is trained for 10 epochs, with the best model saved as best_swin_model.pth based on validation accuracy.

### Testing and Saving Predictions
The script provides functions for loading the best model and making predictions on any dataset (training, validation, or test). Predictions are saved in the following format in CSV files:

image_path: Path to the image.
Columns for each class label representing predicted probabilities.
predicted_class: The predicted class label.

### Usage
Training: Run the script to train the model. This will automatically save the model with the best validation accuracy.

Generate Predictions: Predictions are generated for:

Training data: Saved to train_predictions.csv.
Validation data: Saved to val_predictions.csv.
Test data: Saved to test_predictions.csv.
Saving Predictions: Call the save_predictions_to_csv function, providing the model, dataloader, output CSV file name, and device.

Example:
```bash
save_predictions_to_csv(swin_model, test_loader, 'test_predictions.csv', device)
```


