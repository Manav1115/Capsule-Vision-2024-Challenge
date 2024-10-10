
This repository contains a PyTorch implementation of a Swin Transformer model for image classification, designed to be trained on a custom dataset organized in specific directories (training and validation with subdirectories for each class). To use this code, clone the repository, install the necessary packages using pip install torch torchvision timm matplotlib seaborn scikit-learn, prepare your dataset according to the specified structure, and run the training script (python swin.py). The model utilizes an Adam optimizer with a learning rate of 0.0001 and implements early stopping to prevent overfitting. Upon completion of training, the model evaluates performance by generating training and validation loss plots, validation accuracy plots, a classification report (including precision, recall, and F1-score), and a confusion matrix visualized using Seaborn. Contributions are welcome, and the project is licensed under the MIT License.


![ChallengeHeader](https://github.com/user-attachments/assets/e75f510b-02a8-4fec-b133-11f4ab6c828d)
# Capsule Vision Challenge 2024: Multi-Class Abnormality Classification for Video Capsule Endoscopy

- [Challenge ArXiv](https://arxiv.org/abs/2408.04940)
- [Challenge github repository](https://github.com/misahub2023/Capsule-Vision-2024-Challenge)
- [Training and Validation Dataset Link](https://figshare.com/articles/dataset/Training_and_Validation_Dataset_of_Capsule_Vision_2024_Challenge/26403469?file=48018562)
- [Testing Dataset Link](https://figshare.com/articles/dataset/Testing_Dataset_of_Capsule_Vision_2024_Challenge/27200664?file=49717386)
- [Sample Report Overleaf](https://www.overleaf.com/read/kwhvpznnbzwb#26d62a)

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
## Sample Scripts for Participants
### Data_loader
The [Data_loader.py](https://github.com/misahub2023/Capsule-Vision-2024-Challenge-/blob/main/sample_codes_for_participants/Data_loader.py) script fetches the data from Figshare, unzips and saves it in the current directory.

```bash
python sample_codes_for_participants/Data_loader.py
```
### Eval_metrics_gen_excel 
The [Eval_metrics_gen_excel.py](https://github.com/misahub2023/Capsule-Vision-2024-Challenge-/blob/main/sample_codes_for_participants/Eval_metrics_gen_excel.py) script contains 2 functions:

#### save_predictions_to_excel
  
The `save_predictions_to_excel` function processes the predicted probabilities for a set of images, determines the most likely class for each image, and then saves the results (including both the predicted probabilities and the predicted classes) to an Excel file.
The function takes 3 parameters:
   - `image_paths`: A list of paths to the images. Each path corresponds to an image that was used for prediction.
   - `y_pred`: A numpy array containing the predicted probabilities for each class. Each row corresponds to an image, and each column corresponds to a class.
   - `output_path`: The file path where the Excel file will be saved.

A sample of the excel file which will be generated using this function is available [here](https://github.com/misahub2023/Capsule-Vision-2024-Challenge-/blob/main/sample%20evaluation%20by%20organizing%20members/VGG16/validation_excel.xlsx)

The generated excel file for the test data is to be submitted through for evaluation. [Check here](#submission-format)
Note: The y_pred array should have the predicted probabilites in the order: `['Angioectasia', 'Bleeding', 'Erosion', 'Erythema', 'Foreign Body', 'Lymphangiectasia', 'Normal', 'Polyp', 'Ulcer', 'Worms']`

#### generate_metrics_report
  
The `generate_metrics_report` function generates all the relevant metrics for evaluating a multi-class classification, including classwise and aggregate specificity, ROC AUC scores, precision-recall scores, sensitivity, and F1 scores. This function can be used to evaluate the performance of a trained model on validation data.

The function takes 2 parameters:
  - y_true: The ground truth multi-class labels in one-hot encoded format.
  - y_pred: The predicted probabilities for each class.

Returns: A JSON string containing the detailed metrics report.

Note: The y_pred and y_true array should have the predicted probabilites in the order: `['Angioectasia', 'Bleeding', 'Erosion', 'Erythema', 'Foreign Body', 'Lymphangiectasia', 'Normal', 'Polyp', 'Ulcer', 'Worms']`

#### Usage
To run this script, you'll need to have the following libraries installed:

1. **NumPy**: For numerical operations and array handling.
   - Install using: `pip install numpy`

2. **Pandas**: For data manipulation and analysis.
   - Install using: `pip install pandas`

3. **Scikit-learn**: For machine learning metrics and utilities.
   - Provides functions such as `classification_report`, `roc_auc_score`, `precision_recall_curve`, `recall_score`, and `f1_score`.
   - Install using: `pip install scikit-learn`

4. **JSON**: For parsing JSON data (included in Python standard library, no installation required).

To install all required libraries, you can use the following command:

```bash
pip install numpy pandas scikit-learn
```
### Evaluate_model
The [Evaluate_model.py](https://github.com/misahub2023/Capsule-Vision-2024-Challenge-/blob/main/sample_codes_for_participants/Evaluate_model.py) script is a sample script which shows the usage of the functions from the [Eval_metrics_gen_excel.py](https://github.com/misahub2023/Capsule-Vision-2024-Challenge-/blob/main/sample_codes_for_participants/Eval_metrics_gen_excel.py) script. A VGG16 model has been evaluated in this script, participants can take inspiration from this for their own submissions.


















  
