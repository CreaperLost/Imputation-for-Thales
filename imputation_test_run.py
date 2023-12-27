# INstalling python 3.9
# https://itslinuxfoss.com/install-python-3-9-ubuntu-22-04/
import pandas as pd
import numpy as np
import autosklearn.classification
from generate_missing_values import generate_mcar_missing_values
from automl import AutoML


breast_data = pd.read_csv('missing_data/breast.csv')
Diabetes_data = pd.read_csv('missing_data/diabetes.csv')


# Find the feature's names.
cardio_cols = [i for i in list(Diabetes_data.columns) if i != 'is_anomaly']
breast_cols = [i for i in list(breast_data.columns) if i != 'is_anomaly']

train_budget_minutes = 120

print('Running the diabetes data')
imputed_diabetes = AutoML(Diabetes_data.loc[:,cardio_cols].to_numpy(), Diabetes_data.loc[:,'is_anomaly'].to_numpy(),train_budget_minutes)


print('Running the breast data')


imputed_breast= AutoML(breast_data.loc[:,breast_cols].to_numpy(), breast_data.loc[:,'is_anomaly'].to_numpy(),train_budget_minutes)


imputed_diabetes.to_csv('imputed_data/diabetes.csv',index=None)

imputed_breast.to_csv('imputed_data/breast.csv',index=None)



