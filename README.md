# Diabetes Prediction using Support Vector Machine (SVM)

## Overview
This project implements a machine learning model using Support Vector Machine (SVM) to predict whether a person has diabetes based on the PIMA Diabetes dataset.

## Dataset
The dataset used in this project is the **PIMA Diabetes Dataset**, which contains medical information such as:
- Pregnancies
- Glucose Level
- Blood Pressure
- Skin Thickness
- Insulin Level
- Body Mass Index (BMI)
- Diabetes Pedigree Function
- Age
- Outcome (0: No diabetes, 1: Has diabetes)

## Dependencies
To run this project, install the required libraries using:
```bash
pip install numpy pandas scikit-learn
```

## Steps to Run the Project

### 1. Import Dependencies
```python
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
```

### 2. Load and Explore the Data
```python
diabetes_dataset = pd.read_csv('diabetes.csv')
print(diabetes_dataset.head())
print(diabetes_dataset.shape)
print(diabetes_dataset.describe())
print(diabetes_dataset['Outcome'].value_counts())
```

### 3. Preprocess the Data
Separate features (X) and labels (Y):
```python
X = diabetes_dataset.drop(columns='Outcome', axis=1)
Y = diabetes_dataset['Outcome']
```

Standardize the data:
```python
scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)
```

### 4. Split Data into Training and Testing Sets
```python
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)
```

### 5. Train the SVM Model
```python
classifier = svm.SVC(kernel='linear')
classifier.fit(X_train, Y_train)
```

### 6. Evaluate Model Accuracy
```python
X_train_prediction = classifier.predict(X_train)
train_accuracy = accuracy_score(X_train_prediction, Y_train)
print("Training Accuracy:", train_accuracy)

X_test_prediction = classifier.predict(X_test)
test_accuracy = accuracy_score(X_test_prediction, Y_test)
print("Testing Accuracy:", test_accuracy)
```

### 7. Make Predictions
To predict whether a person has diabetes:
```python
input_data = (1, 89, 66, 23, 94, 28.1, 0.167, 21)
input_data_as_nparray = np.asarray(input_data).reshape(1, -1)
std_data = scaler.transform(input_data_as_nparray)

prediction = classifier.predict(std_data)
if prediction[0] == 1:
    print("The person has diabetes.")
else:
    print("The person does not have diabetes.")
```

## Results
- Training Accuracy: ~78.66%
- Testing Accuracy: ~77.27%

## Conclusion
This project demonstrates how to use **SVM (Support Vector Machine)** for binary classification in medical diagnosis. The model provides a decent accuracy and can be improved further with hyperparameter tuning or additional data.

## License
This project is for educational purposes. Feel free to modify and improve it!

