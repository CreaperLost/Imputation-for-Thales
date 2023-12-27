
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
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score,roc_auc_score,f1_score
from sklearn.model_selection import StratifiedShuffleSplit

def get_best_model(features, best_config):
    configuration = best_config['feature_preprocessor:__choice__']
    scaler = StandardScaler()
    if configuration == 'MeanModePreprocessor':
        imputor = MeanModePreprocessor()
    elif configuration == 'MissForestProcessor':
        depth_config = best_config['feature_preprocessor:MissForestProcessor:max_depth']
        print(f'Depth : {depth_config}')
        imputor = MissForestProcessor(max_depth=depth_config)
    elif configuration == 'DAEProcessor':
        dropout = best_config['feature_preprocessor:DAEProcessor:dropout']
        theta = best_config['feature_preprocessor:DAEProcessor:theta']
        imputor = DAEProcessor(dropout= dropout, theta = theta)
         
    scaled_features = scaler.fit_transform(features)
    scaled_imputed_features = imputor.fit_transform(scaled_features)
    imputed_features = scaler.inverse_transform(scaled_imputed_features)
    return imputed_features


def AutoML(features, outcome, time):
    autosklearn.pipeline.components.feature_preprocessing.add_preprocessor(MeanModePreprocessor)
    autosklearn.pipeline.components.data_preprocessing.add_preprocessor(ScalingPreprocessing)
    autosklearn.pipeline.components.feature_preprocessing.add_preprocessor(MissForestProcessor)
    autosklearn.pipeline.components.feature_preprocessing.add_preprocessor(DAEProcessor)

    accuracy_scorer = autosklearn.metrics.make_scorer(
    name="auc",
    score_func=roc_auc_score,
    optimum=1,
    greater_is_better=True,
    needs_proba=False,
    needs_threshold=False,
    )

    # Below two flags are provided to speed up calculations
    # Not recommended for a real implementation
    # smac_scenario_args={"runcount_limit": 5},
    time_in_seconds = time * 60

    logging_config = {
    "version": 1,
    "disable_existing_loggers": True,
    "formatters": {
        "custom": {
            # More format options are available in the official
            # `documentation <https://docs.python.org/3/howto/logging-cookbook.html>`_
            "format": "" # %(asctime)s - %(name)s - %(levelname)s - %(message)s
        }
    },
    # Any INFO level msg will be printed to the console
    "handlers": {
        "console": {
            "level": "ERROR",
            "formatter": "custom",
            "class": "logging.StreamHandler",
            "stream": "ext://sys.stdout",
        },
    },
    "loggers": {
        "": {  # root logger
            "level": "ERROR",
        },
        "Client-EnsembleBuilder": {
            "level": "ERROR",
            "handlers": ["console"],
        },
    },
    }

    clf = AutoSklearnClassifier(
        time_left_for_this_task=time_in_seconds,
        include={"data_preprocessor": ["ScalingPreprocessing"],
                'feature_preprocessor' : ['MeanModePreprocessor','MissForestProcessor','DAEProcessor'] #'MissForestProcessor','DAEProcessor','MeanModePreprocessor',,'MissForestProcessor'
                },
        initial_configurations_via_metalearning=0,
        ensemble_kwargs = {'ensemble_size': 1},
        resampling_strategy = 'cv',
        resampling_strategy_arguments ={"n_folds" :10},
        metric = accuracy_scorer,
        memory_limit= 8000,
        logging_config=logging_config,
        seed =1 ,
    )    

    clf.fit(features,outcome)

    print(clf.sprint_statistics())


    losses_and_configurations = [
        (run_value.cost, run_key.config_id)
        for run_key, run_value in clf.automl_.runhistory_.data.items()
    ]
    losses_and_configurations.sort()
    print("Lowest loss:", losses_and_configurations[0][0])
    best= clf.automl_.runhistory_.ids_config[losses_and_configurations[0][1]].get_dictionary()
    print("Best configuration:",best)
    print(best['feature_preprocessor:__choice__'])
    
    imputed_features = get_best_model(features,best)
    imputed_merged_data = pd.concat([pd.DataFrame(imputed_features),pd.Series(outcome,name='is_anomaly')],axis=1)
    return imputed_merged_data

    """for run_key in clf.automl_.runhistory_.data:
    print("#########")
    print(run_key)
    print(clf.automl_.runhistory_.data[run_key])"""








