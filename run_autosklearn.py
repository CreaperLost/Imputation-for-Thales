
import pandas as pd
import numpy as np
import autosklearn 
from autosklearn.classification import AutoSklearnClassifier

from imputation_code.mm_preprocessor import MeanModePreprocessor
from imputation_code.dae_preprocessor import DAEProcessor
from imputation_code.mf_preprocessor import MissForestProcessor
from imputation_code.scaler import ScalingPreprocessing
from imputation_code.no_preprocessor import NoPreprocessing
from imputation_code.dae_preprocessor import DAEProcessor
from imputation_code.mf_preprocessor import MissForestProcessor
import autosklearn.classification
from sklearn.metrics import accuracy_score


def AutoML(features, outcome):
    autosklearn.pipeline.components.feature_preprocessing.add_preprocessor(MeanModePreprocessor)
    autosklearn.pipeline.components.data_preprocessing.add_preprocessor(ScalingPreprocessing)
    autosklearn.pipeline.components.feature_preprocessing.add_preprocessor(MissForestProcessor)
    autosklearn.pipeline.components.feature_preprocessing.add_preprocessor(DAEProcessor)


    accuracy_scorer = autosklearn.metrics.make_scorer(
    name="accu",
    score_func=accuracy_score,
    optimum=1,
    greater_is_better=True,
    needs_proba=False,
    needs_threshold=False,
    )

    # Below two flags are provided to speed up calculations
    # Not recommended for a real implementation
    # smac_scenario_args={"runcount_limit": 5},

    clf = AutoSklearnClassifier(
        time_left_for_this_task=7200,
        include={"data_preprocessor": ["ScalingPreprocessing"],
                'feature_preprocessor' : ['MissForestProcessor','DAEProcessor','MeanModePreprocessor'] #
                },
        initial_configurations_via_metalearning=0,
        ensemble_kwargs = {'ensemble_size': 1},
        resampling_strategy = 'cv',
        resampling_strategy_arguments ={"folds" :5},
        metric = accuracy_scorer,
        memory_limit= 8000
    )    

    clf.fit(features,outcome)

    print(clf.sprint_statistics())


    losses_and_configurations = [
        (run_value.cost, run_key.config_id)
        for run_key, run_value in clf.automl_.runhistory_.data.items()
    ]
    losses_and_configurations.sort()
    print("Lowest loss:", losses_and_configurations[0][0])
    print(
        "Best configuration:",
        clf.automl_.runhistory_.ids_config[losses_and_configurations[0][1]],
    )

    """for run_key in clf.automl_.runhistory_.data:
    print("#########")
    print(run_key)
    print(clf.automl_.runhistory_.data[run_key])"""








