# INstalling python 3.9
# https://itslinuxfoss.com/install-python-3-9-ubuntu-22-04/
import pandas as pd
import numpy as np
import autosklearn.classification
from generate_missing_values import generate_mcar_missing_values
from run_autosklearn import AutoML


breast_data = pd.read_csv('missing_datasets/breast.csv')
Diabetes_data = pd.read_csv('missing_datasets/diabetes.csv')


# Find the feature's names.
cardio_cols = [i for i in list(Diabetes_data.columns) if i != 'is_anomaly']
breast_cols = [i for i in list(breast_data.columns) if i != 'is_anomaly']

print('Running the diabetes data')
AutoML(Diabetes_data.loc[:,cardio_cols].to_numpy(), Diabetes_data.loc[:,'is_anomaly'].to_numpy(),5)


print('Running the breast data')


AutoML(breast_data.loc[:,breast_cols].to_numpy(), breast_data.loc[:,'is_anomaly'].to_numpy(),5)








