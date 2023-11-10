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
from imputation_code.mm_preprocessor import MeanModePreprocessor
from imputation_code.dae_preprocessor import DAEProcessor
from imputation_code.mf_preprocessor import MissForestProcessor


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


imputation_methods = [
    {
        'name':'mean',
        'model': MeanModePreprocessor,
        'params': []
    },
    {
        'name':'missforest',
        'model': MissForestProcessor,
        'params': [10, 20]
    },
    {
        'name':'dae',
        'model': DAEProcessor,
        'params': [(0.25, 5) ,(0.25, 7), (0.25, 10),
                   (0.4, 5)  ,(0.4, 7),  (0.4, 10),
                   (0.5, 5)  ,(0.5, 7),  (0.5, 10)]
    }
]

# Define the modeling algorithms and their hyperparameters for GridSearchCV
models = [
    {
        'name': 'RandomForest',
        'model': RandomForestClassifier,
        'params': [
            (100,10), (100,20), (100,None)
        ]
    },
    {
        'name': 'SVM',
        'model': SVC,
        'params': [
            ('linear',0.1),('linear',1),('linear',10),
            ('rbf',0.1),('rbf',1),('rbf',10),
        ]
    }
]


def find_imputation_model(imp_name, imp_model, imp_param):
    if imp_name == 'mean':
        return imp_model()
    elif imp_name == 'missforest':
        return imp_model(max_depth = imp_param)
    else:
        return imp_model(parameters = {'dropout':imp_param[0], 'theta':imp_param[1]})

# Create a pipeline with preprocessing steps and modeling algorithm
results = []

for imp_method in imputation_methods:
    imp_name = imp_method['name']
    imp_model = imp_method['model']
    imp_params = imp_params['params']

    for imp_param in imp_params:

        imp = find_imputation_model(imp_name, imp_model, imp_params)
        print(f'{imp_name, imp_model, imp_params,imp}')

        for model_info in models:

            model = model_info['model']
            model_params = model_info['params']
            model_name = model_info['name']

            for model_param in model_params:
                    
                imp = imp_model(model_name, model, imp_param)
                # Create a pipeline
                pipeline = Pipeline([
                            ('scaler', scaler),
                            ('imputation',imp)
                            ('model', model)])


# Print the results
for result in results:
    print(f"Model: {result['name']}")
    print(f"Best Training Accuracy: {result['best_score']:.4f}")
    print(f"Test Accuracy: {result['test_score']:.4f}\n")
