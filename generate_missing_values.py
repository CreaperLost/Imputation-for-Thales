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

