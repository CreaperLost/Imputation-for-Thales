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
    data_shape = data.shape

    mask = (np.random.rand(data_shape[0],data_shape[1]) < percentage)

    data[mask] = np.nan
    return data

