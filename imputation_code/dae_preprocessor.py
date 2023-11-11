from autosklearn.pipeline.components.base import AutoSklearnPreprocessingAlgorithm
from autosklearn.pipeline.constants import SPARSE, DENSE, UNSIGNED_DATA, INPUT
from autosklearn.askl_typing import FEAT_TYPE_TYPE
from ConfigSpace.configuration_space import ConfigurationSpace
from typing import Optional
from pprint import pprint
import autosklearn.classification
from sklearn.impute import SimpleImputer
from imputation_code.dae import DAE
from ConfigSpace.hyperparameters import (
    UniformIntegerHyperparameter,
    CategoricalHyperparameter,
)
import pandas as pd

class DAEProcessor(AutoSklearnPreprocessingAlgorithm):
    def __init__(self,dropout,theta, **kwargs):
        self.dropout = dropout
        self.theta = theta
        """This preprocessors does not change the data"""
        # Some internal checks makes sure parameters are set
        for key, val in kwargs.items():
            setattr(self, key, val)

    def fit(self, X, Y=None):
        self.imputer = DAE(parameters={'dropout':self.dropout, 'theta': self.theta}).fit(X)
        return self

    def transform(self, X):
        
        X_new = self.imputer.transform(X)
        
        return X_new

    @staticmethod
    def get_properties(dataset_properties=None):
        return {
            "shortname": "DAEProcessor",
            "name": "DAE Processor",
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
        cs = ConfigurationSpace()
        theta = CategoricalHyperparameter(name="theta", choices=[5,7,10])
        dropout = CategoricalHyperparameter(name="dropout", choices=[0.25,0.4,0.5])
        cs.add_hyperparameters([theta,dropout])
        return cs  # Return an empty configuration as there is None