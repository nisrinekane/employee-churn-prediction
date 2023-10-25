from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV

def train_xgboost_with_gridsearch(X_train, y_train):
    # Define the model
    model = XGBClassifier()

    # Define parameters to search
    param_grid = {
        'learning_rate': [0.01, 0.05, 0.1],
        'n_estimators': [100, 500, 1000],
        'max_depth': [3, 5, 7, 9],
        'subsample': [0.8, 0.9, 1.0],
        'colsample_bytree': [0.8, 0.9, 1.0],
    }

    # Grid search
    grid_search = GridSearchCV(model, param_grid, cv=3, scoring='accuracy')
    grid_search.fit(X_train, y_train)

    # Return the best model
    return grid_search.best_estimator_
