from data_preprocessing import preprocess_data
from train_model import train_xgboost_with_gridsearch
from model_evaluation import evaluate_model
from sklearn.feature_selection import RFECV
import pandas as pd
from sklearn.linear_model import LogisticRegression


# preprocess data:
X_train, X_test, y_train, y_test, column_names = preprocess_data('HR-Employee-Attrition.csv')

# Feature selection using Recursive Feature Elimination with Cross-Validation (RFECV)
# Using a simpler Logistic Regression model for feature selection for the sake of speed
selector = RFECV(estimator=LogisticRegression(max_iter=1000), step=1, cv=5)
X_train_selected = selector.fit_transform(X_train, y_train)
X_test_selected = selector.transform(X_test)

# train model:
model = train_xgboost_with_gridsearch(X_train_selected, y_train)

# evaluate model:
accuracy, precision, recall, f1 = evaluate_model(model, X_test_selected, y_test)

print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1: {f1}")

# To display the features that were selected:
selected_features = pd.Series(selector.support_, index=column_names)
print("Selected Features:\n", selected_features[selected_features].index.tolist())
