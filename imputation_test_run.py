
import pandas as pd
import numpy as np
import autosklearn 
from autosklearn.classification import AutoSklearnClassifier

from imputation_code.mm_preprocessor import MeanModePreprocessor
from imputation_code.scaler import ScalingPreprocessing
from imputation_code.no_preprocessor import NoPreprocessing
from imputation_code.dae_preprocessor import DAEProcessor
from imputation_code.mf_preprocessor import MissForestProcessor
import autosklearn.classification
from generate_missing_values import generate_mcar_missing_values
from sklearn.metrics import accuracy_score

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
Diabetes_data.loc[:,cardio_cols] = generate_mcar_missing_values(Diabetes_data.loc[:,cardio_cols], 0.25).copy()

breast_data.loc[:,breast_cols] = generate_mcar_missing_values(breast_data.loc[:,breast_cols], 0.25).copy()

"""model = MissForestProcessor(10).fit(Diabetes_data.loc[:,cardio_cols])
imp = model.transform(Diabetes_data.loc[:,cardio_cols])

print(imp)

model = DAEProcessor(0.25,5).fit(Diabetes_data.loc[:,cardio_cols])
imp = model.transform(Diabetes_data.loc[:,cardio_cols])

print(imp)"""

#autosklearn.pipeline.components.data_preprocessing.add_preprocessor(NoPreprocessing)
# Add NoPreprocessing component to auto-sklearn.
#autosklearn.pipeline.components.feature_preprocessing.add_preprocessor(MeanModePreprocessor)
autosklearn.pipeline.components.data_preprocessing.add_preprocessor(ScalingPreprocessing)
#autosklearn.pipeline.components.feature_preprocessing.add_preprocessor(MissForestProcessor)
autosklearn.pipeline.components.feature_preprocessing.add_preprocessor(DAEProcessor)


accuracy_scorer = autosklearn.metrics.make_scorer(
    name="accu",
    score_func=accuracy_score,
    optimum=1,
    greater_is_better=True,
    needs_proba=False,
    needs_threshold=False,
)
#'MissForestProcessor'

#"data_preprocessor": ['ScalingPreprocessing'],\

#'feature_preprocessor':['MeanModePreprocessor','MissForestProcessor']
clf = AutoSklearnClassifier(
    time_left_for_this_task=300,
    include={"data_preprocessor": ["ScalingPreprocessing"],
             'feature_preprocessor' : ['DAEProcessor']
             },
    # Bellow two flags are provided to speed up calculations
    # Not recommended for a real implementation
    initial_configurations_via_metalearning=0,
    smac_scenario_args={"runcount_limit": 10},
    ensemble_kwargs = {'ensemble_size': 1},
    resampling_strategy = 'cv',
    resampling_strategy_arguments ={"folds" :5},
)    #metric = accuracy_scorer,
clf.fit(Diabetes_data.loc[:,cardio_cols], Diabetes_data.loc[:,'is_anomaly'])


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




for run_key in clf.automl_.runhistory_.data:
    print("#########")
    print(run_key)
    print(clf.automl_.runhistory_.data[run_key])




