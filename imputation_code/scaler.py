from autosklearn.pipeline.components.base import AutoSklearnPreprocessingAlgorithm
from autosklearn.pipeline.constants import SPARSE, DENSE, UNSIGNED_DATA, INPUT
from autosklearn.askl_typing import FEAT_TYPE_TYPE
from ConfigSpace.configuration_space import ConfigurationSpace
from typing import Optional
from pprint import pprint
import autosklearn.classification
from sklearn.preprocessing import StandardScaler
import pandas as pd

class ScalingPreprocessing(AutoSklearnPreprocessingAlgorithm):
    def __init__(self, **kwargs):
        """This preprocessors does not change the data"""
        # Some internal checks makes sure parameters are set
        for key, val in kwargs.items():
            setattr(self, key, val)

    def fit(self, X, Y=None):
        self.scaler = StandardScaler()
        self.scaler.fit(X)
        
        return self

    def transform(self, X):
        X_new = self.scaler.transform(X)
        return X_new

    @staticmethod
    def get_properties(dataset_properties=None):
        return {
            "shortname": "ScalingPreprocessor",
            "name": "ScalingPreprocessor",
            "handles_regression": True,
            "handles_classification": True,
            "handles_multiclass": True,
            "handles_multilabel": True,
            "handles_multioutput": True,
            "is_deterministic": True,
            "input": (SPARSE, DENSE, UNSIGNED_DATA),
            "output": (INPUT,),
        }

    @staticmethod
    def get_hyperparameter_search_space(
        feat_type: Optional[FEAT_TYPE_TYPE] = None, dataset_properties=None
    ):
        return ConfigurationSpace()  # Return an empty configuration as there is None
