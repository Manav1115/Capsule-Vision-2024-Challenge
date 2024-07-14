import argparse
from tensorflow.keras.models import load_model
import numpy as np
from Eval_metrics_gen_excel import save_predictions_to_excel,generate_metrics_report
import pandas as pd
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.vgg16 import preprocess_input
import os

model_path="sample evaluation by organizing members/model.keras"
#this only works for .keras and .h5 models
model = load_model(model_path)
#modify according to the model type being used

#these data loading functions are specific to VGG16, modify accordingly 
def load_and_preprocess_image(full_path, target_size):
    img = load_img(full_path, target_size=target_size)
    img_array = img_to_array(img)
    preprocessed_img = preprocess_input(img_array)
    return preprocessed_img

def get_data(excel_path, base_dir, image_size=(32, 32)):
    df = pd.read_excel(excel_path)
    df = df.dropna(subset=['image_path'])
    class_columns = ['Angioectasia', 'Bleeding', 'Erosion', 'Erythema', 'Foreign Body', 'Lymphangiectasia', 'Normal', 'Polyp', 'Ulcer', 'Worms']
    X = np.array([load_and_preprocess_image(os.path.join(base_dir, path), image_size) for path in df['image_path'].values])
    y = df[class_columns].values
    return X, y, df

def load_test_data(test_dir, image_size=(32, 32)):
    image_paths = [os.path.join(test_dir, fname) for fname in os.listdir(test_dir) if fname.lower().endswith(('jpg'))]
    X_test = np.array([load_and_preprocess_image(path, image_size) for path in image_paths])
    return X_test, image_paths

#these parameters are also specific to the sample being shown here and can be changed
base_dir="D:/misahub/Dataset/Dataset"
val_excel_path="D:/misahub/Dataset/Dataset/validation/validation_data.xlsx"
image_size=(32,32)
test_dir="D:/misahub/Test set for participants/Final test set/Images"

X_val, y_val, val_df = get_data(val_excel_path, base_dir=base_dir, image_size=image_size)
y_val_pred = model.predict(X_val)
#this function generates all the necessary metrics to be used for evaluation
#ensure that the order of columns in y_val and y_val_pred is mantained before use
df=generate_metrics_report(y_val,y_val_pred)
print(df)

output_val_predictions="validation_excel.xlsx"
save_predictions_to_excel(val_df['image_path'].values, y_val_pred, output_val_predictions)

X_test, test_image_paths = load_test_data(test_dir, image_size)

y_test_pred = model.predict(X_test)
output_test_predictions="test_excel.xlsx"
save_predictions_to_excel(test_image_paths, y_test_pred, output_test_predictions)
