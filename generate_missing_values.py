import pandas as pd
import numpy as np

def generate_mcar_missing_values(data: pd.DataFrame, percentage: float) -> pd.DataFrame:
    """
    Input : 
    data: dataset without missing values
    percentage: float that specifies missingness percentage in [0,1]

    Output:
    A dataframe with missing values at specified misisngness percentage.
    """
    new_data = data.copy()

    data_shape = new_data.shape

    mask = (np.random.rand(data_shape[0],data_shape[1]) <= percentage)
  
    new_data.mask(mask, np.nan,inplace=True)
    
    return new_data

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
Diabetes_data.loc[:,cardio_cols] = generate_mcar_missing_values(Diabetes_data.loc[:,cardio_cols], 0.1).copy()

breast_data.loc[:,breast_cols] = generate_mcar_missing_values(breast_data.loc[:,breast_cols], 0.1).copy()


Diabetes_data.to_csv('missing_data/diabetes.csv',index=None)
breast_data.to_csv('missing_data/breast.csv',index=None)
