import pandas as pd
import numpy as np

breast_data = pd.read_csv('datasets/breast-w.csv')

cardio_data = pd.read_csv('datasets/Cardiotocography.csv')

breast_data.head(5)

cardio_data.head(5)

from generate_missing_values import generate_mcar_missing_values


breast_data_missing = generate_mcar_missing_values(breast_data, 0.25)
print(breast_data_missing.isna().sum().sum()/(breast_data_missing.shape[0]*breast_data_missing.shape[1]))


cardio_data_missing = generate_mcar_missing_values(cardio_data, 0.25)
print(cardio_data_missing.isna().sum().sum()/(cardio_data_missing.shape[0]*cardio_data_missing.shape[1]))


