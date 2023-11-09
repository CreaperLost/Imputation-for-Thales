from autosklearn.pipeline.components.base import AutoSklearnPreprocessingAlgorithm
from autosklearn.pipeline.constants import SPARSE, DENSE, UNSIGNED_DATA, INPUT
from autosklearn.askl_typing import FEAT_TYPE_TYPE
from ConfigSpace.configuration_space import ConfigurationSpace
from typing import Optional
from pprint import pprint
import autosklearn.classification
from sklearn.impute import SimpleImputer
from imputation_code.mf import IterativeImputer
#from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor
from ConfigSpace.hyperparameters import (
    UniformIntegerHyperparameter,
    CategoricalHyperparameter,
)

class MissForestProcessor(AutoSklearnPreprocessingAlgorithm):
    def __init__(self,max_depth, **kwargs):
        self.max_depth = max_depth
        """This preprocessors does not change the data"""
        # Some internal checks makes sure parameters are set
        for key, val in kwargs.items():
            setattr(self, key, val)

    def fit(self, X, Y=None):
        self.imputer = IterativeImputer(parameters={'max_depth':self.max_depth}).fit(X)
        return self

    def transform(self, X):
        X = self.imputer.transform(X)
        return X

    @staticmethod
    def get_properties(dataset_properties=None):
        return {
            "shortname": "NoPreprocessing",
            "name": "NoPreprocessing",
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
        solver = CategoricalHyperparameter(name="max_depth", choices=[10,20])
        cs.add_hyperparameters([solver])
        return cs  # Return an empty configuration as there is None