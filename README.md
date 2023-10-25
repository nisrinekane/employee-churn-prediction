# Employee Churn Prediction

This project aims to predict employee churn based on various employee attributes using IBM's HR EMployee Attrition dataset. 
We XGBoost to predict whether an employee will leave the company or not.

## Overview:

### Data Preprocessing:
The dataset is loaded and processed to convert categorical variables into a format suitable for machine learning.
Features are scaled to ensure they contribute equally to model training.

### Feature Engineering:
New features such as 'AgeGroup' and 'HighDistance' were derived from existing attributes to improve the model's predictive score.

### Feature Selection:
Recursive Feature Elimination with Cross-Validation (RFECV) was used to identify the most predictive features.

### Model Training:
XGBoost was used as classifier, optimizing its hyperparameters through grid search with cross-validation.

### Model Evaluation:
The model's performance is evaluated on a separate test set and various metrics (accuracy, precision, recall, F1 score) are calculated to understand its effectiveness.

## Setup:
### Dependencies:
To install the required libraries, navigate to the project's root directory and run:
```
pip install -r requirements.txt
```

### Running the Model:
Execute the main.py script: 
```
python3 main.py
```

The script will preprocess the data, select features, train the model, and then evaluate its performance.
