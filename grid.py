import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
import pandas as pd
from generate_missing_values import generate_mcar_missing_values


breast_data = pd.read_csv('datasets/breast-w.csv')
Diabetes_data = pd.read_csv('datasets/Diabetes.csv')


breast_data.head(5)
breast_data.drop(columns=['subspaces'],inplace=True)
breast_data


Diabetes_data.head(5)
Diabetes_data.drop(columns=['subspaces'],inplace=True)
Diabetes_data


cardio_cols = [i for i in list(Diabetes_data.columns) if i != 'is_anomaly']
breast_cols = [i for i in list(breast_data.columns) if i != 'is_anomaly']

#Only inser missing data at features!
Diabetes_data.loc[:,cardio_cols] = generate_mcar_missing_values(Diabetes_data.loc[:,cardio_cols], 0.25).copy()

breast_data.loc[:,breast_cols] = generate_mcar_missing_values(breast_data.loc[:,breast_cols], 0.25).copy()



# Define the preprocessing steps
scaler = StandardScaler()
feature_selector = SelectKBest(f_classif, k=2)

# Define the modeling algorithms and their hyperparameters for GridSearchCV
models = [
    {
        'name': 'RandomForest',
        'model': RandomForestClassifier(),
        'params': {
            'model__n_estimators': [100],
            'model__max_depth': [None, 10, 20]
        }
    },
    {
        'name': 'GradientBoosting',
        'model': GradientBoostingClassifier(),
        'params': {
            'model__n_estimators': [50, 100, 200],
            'model__learning_rate': [0.01, 0.1, 0.2]
        }
    },
    {
        'name': 'SVM',
        'model': SVC(),
        'params': {
            'model__C': [0.1, 1, 10],
            'model__kernel': ['linear', 'rbf']
        }
    }
]

# Create a pipeline with preprocessing steps and modeling algorithm
results = []
for model_info in models:
    model = model_info['model']
    params = model_info['params']
    name = model_info['name']

    # Create a pipeline
    pipeline = Pipeline([
        ('scaler', scaler),
        ('feature_selector', feature_selector),
        ('model', model)
    ])

    # Perform hyperparameter optimization using GridSearchCV
    grid_search = GridSearchCV(pipeline, params, cv=5, n_jobs=-1)
    grid_search.fit(X_train, y_train)

    # Store the best model and its performance
    best_model = grid_search.best_estimator_
    best_score = grid_search.best_score_
    test_score = best_model.score(X_test, y_test)

    results.append({
        'name': name,
        'best_model': best_model,
        'best_score': best_score,
        'test_score': test_score
    })

# Print the results
for result in results:
    print(f"Model: {result['name']}")
    print(f"Best Training Accuracy: {result['best_score']:.4f}")
    print(f"Test Accuracy: {result['test_score']:.4f}\n")
