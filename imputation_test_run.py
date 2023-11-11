
import pandas as pd
import numpy as np
import autosklearn.classification
from generate_missing_values import generate_mcar_missing_values
from run_autosklearn import AutoML



breast_data = pd.read_csv('datasets/breast-w.csv')
Diabetes_data = pd.read_csv('datasets/Diabetes.csv')


breast_data.head(5)
breast_data.drop(columns=['subspaces'],inplace=True)
breast_data


Diabetes_data.head(5)
Diabetes_data.drop(columns=['subspaces'],inplace=True)
Diabetes_data

#Insert missing values to both datasets!

# Find the feature's names.
cardio_cols = [i for i in list(Diabetes_data.columns) if i != 'is_anomaly']
breast_cols = [i for i in list(breast_data.columns) if i != 'is_anomaly']

#Only inser missing data at features!
Diabetes_data.loc[:,cardio_cols] = generate_mcar_missing_values(Diabetes_data.loc[:,cardio_cols], 0.25).copy()

breast_data.loc[:,breast_cols] = generate_mcar_missing_values(breast_data.loc[:,breast_cols], 0.25).copy()

AutoML(Diabetes_data.loc[:,cardio_cols].to_numpy(), Diabetes_data.loc[:,'is_anomaly'].to_numpy())

AutoML(breast_data.loc[:,breast_cols].to_numpy(), breast_data.loc[:,'is_anomaly'].to_numpy())








