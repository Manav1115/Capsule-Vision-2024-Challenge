


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

This repository contains a PyTorch implementation of a Swin Transformer model for image classification, designed to be trained on a custom dataset organized in specific directories (training and validation with subdirectories for each class). To use this code, clone the repository, install the necessary packages using pip install torch torchvision timm matplotlib seaborn scikit-learn, prepare your dataset according to the specified structure, and run the training script (python swin.py). The model utilizes an Adam optimizer with a learning rate of 0.0001 and implements early stopping to prevent overfitting. Upon completion of training, the model evaluates performance by generating training and validation loss plots, validation accuracy plots, a classification report (including precision, recall, and F1-score), and a confusion matrix visualized using Seaborn. Contributions are welcome, and the project is licensed under the MIT License.

















  
