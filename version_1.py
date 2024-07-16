# V1
# Meta Data Accuracy:         0.9302325581395349
# Training Data Accuracy:     0.8072126297533729
# Test Data Accuracy:         0.8220902612826604

# -------------------------------------------------------

import cv2
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd       
from sklearn.metrics import confusion_matrix, accuracy_score                                                                            
from skimage.feature import hog                             
from skimage.filters import gaussian                 
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split  

# Subset 1: Meta Data
X_meta = []             # Stores meta images
y_meta = []             # Stores meta labels
y_meta_color = []       # Stores meta data color

# Subset 2: Partial Training Data
X = []                  # Stores training images
y = []                  # Stores classID  
y_color_train = []      # Stores colorID

# Test Data
X_test_read = []        # Stores test images 
y_test_class = []       # Stores class labels of test images
y_color_test = []       # Stores test data color

# Processed Image List
X_processed = []        # Stores processed training images
X_test_processed = []   # Stores processed test images

# Dictionary to map labels to their colors 
meta_dict = {}  

# Read Data ====================================================

# Store meta data
meta_data = pd.read_csv("data/Meta.csv")
for index, row in meta_data.iterrows():
    img_path = row['Path']
    img_path = "data/" + img_path
    img = cv2.imread(img_path)
    label = row['ClassId']
    color = row['ColorId']
    # shape = row['ShapeId']
    if img is not None:
        X_meta.append(img)
        y_meta.append(label)
        y_meta_color.append(color)
        # y_meta_shape.append(shape)
        meta_dict[label] = (color)

# Function to map classID to colorID (use for test data)
color_mapping = {row['ClassId']: row['ColorId'] for index, row in meta_data.iterrows()}

# Function to read CSV and load images
def read_csv(csv_file_path, X1, y1, color_list):
    data = pd.read_csv(csv_file_path)
    for index, row in data.iterrows():
        img_path = row['Path']
        label = row['ClassId']
        color_id = color_mapping[label]  # Get the color ID from the mapping
        roi_x1 = row['Roi.X1']
        roi_y1 = row['Roi.Y1']
        roi_x2 = row['Roi.X2']
        roi_y2 = row['Roi.Y2']
        img = cv2.imread("data/" + img_path)
        # Crop images based on the given region of interest (ROI). 
        if img is not None:
            cropped_img = img[roi_y1:roi_y2, roi_x1:roi_x2]
            X1.append(cropped_img)
            y1.append(label)
            color_list.append(color_id)

# Training and test data 
read_csv("data/Train.csv", X, y, y_color_train)
read_csv("data/Test.csv", X_test_read, y_test_class, y_color_test)

# Extract Dominant Color ===================================================

def extract_color_id(image):
    
    # Convert to HSV color space
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Equalize the v-channel
    h, s, v = cv2.split(image_hsv)
    v = cv2.equalizeHist(v)
    image_hsv = cv2.merge((h, s, v))

    # cv2.imwrite("hsv.png", image_hsv)
    
    # Define the color ranges
    red_lower = (0, 100, 100)
    red_upper = (10, 255, 255)
    blue_lower = (100, 150, 0)
    blue_upper = (140, 255, 255)
    yellow_lower = (20, 100, 100)
    yellow_upper = (30, 255, 255)
    white_lower = (0, 0, 200)
    white_upper = (180, 25, 255)
    
    # Create masks for each color
    mask_red = cv2.inRange(image_hsv, red_lower, red_upper)
    mask_blue = cv2.inRange(image_hsv, blue_lower, blue_upper)
    mask_yellow = cv2.inRange(image_hsv, yellow_lower, yellow_upper)
    mask_white = cv2.inRange(image_hsv, white_lower, white_upper)
    
    # Count the number of pixels in each mask
    red_count = cv2.countNonZero(mask_red)            # 0 ColorID
    blue_count = cv2.countNonZero(mask_blue)          # 1 ColorID
    yellow_count = cv2.countNonZero(mask_yellow)      # 2 ColorID
    white_count = cv2.countNonZero(mask_white)        # 3 ColorID
    
    # Determine the dominant color and map to color ID
    sum_pixels =red_count + blue_count + yellow_count + white_count
    color_counts = [red_count, blue_count, yellow_count]
    if max(color_counts) < ((sum_pixels)/98):
        color_id =  3   # If red, blue, and yellow count too low, default to white as dominant color
    else:
        color_id = color_counts.index(max(color_counts))
    return color_id

# Test accuracy for meta data
length = len(X_meta):
for i in range(length):
    print("Label", y_meta[i])
    print("Actual", y_meta_color[i])
    extracted = extract_color_id(X_meta[i])
    print("Predicted", extracted)

# Test accuracy for train data
length = len(y)
for i in range(length):
    print("Label", y[i])
    print("Actual", y_color_train[i])
    extracted = extract_color_id(X[i])
    print("Predicted", extracted)

# Test accuracy for test data
length = len(X_test_read)
for i in range(length):
    print("Label", y_test_class[i])
    print("Actual", y_color_test[i])
    extracted = extract_color_id(X_test_read[i])
    print("Predicted", extracted)

# Evaluate accuracy and create confusion matrices ===================================================

def evaluate_and_confusion_matrix(X_data, y_actual_color):
    y_pred_color = [extract_color_id(img) for img in X_data]
    accuracy = accuracy_score(y_actual_color, y_pred_color)
    conf_matrix = confusion_matrix(y_actual_color, y_pred_color)
    
    return accuracy, conf_matrix
    
# Evaluate accuracy for meta data
meta_accuracy, meta_conf_matrix = evaluate_and_confusion_matrix(X_meta, y_meta_color)

# Evaluate accuracy for training data
train_accuracy, train_conf_matrix = evaluate_and_confusion_matrix(X, y_color_train)

# Evaluate accuracy for test data
test_accuracy, test_conf_matrix = evaluate_and_confusion_matrix(X_test_read, y_color_test)

# Print accuracies
print(f"Meta Data Accuracy: {meta_accuracy}")
print(f"Training Data Accuracy: {train_accuracy}")
print(f"Test Data Accuracy: {test_accuracy}")

# Plot and save confusion matrices
def plot_confusion_matrix(cm, title, save_path=None):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["Red", "Blue", "Yellow", "White"], yticklabels=["Red", "Blue", "Yellow", "White"])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(title)
    
    if save_path:
        plt.savefig(save_path)
        print(f"Confusion matrix saved to {save_path}")
    
    plt.show()

plot_confusion_matrix(meta_conf_matrix, 'Confusion Matrix for Meta Data', 'meta_conf_matrix.png')
plot_confusion_matrix(train_conf_matrix, 'Confusion Matrix for Training Data', 'train_conf_matrix.png')
plot_confusion_matrix(test_conf_matrix, 'Confusion Matrix for Test Data', 'test_conf_matrix.png')


