# Documentation for Imputation

The imputation model is responsible for filling in the missing data by selecting the best imputation method, that returns the best autosklearn final predictive performance.

The imputation algorithms included:

1.  Denoise Auto-Encoder imputation
2.  MissForest imputation
3.  Mean-Mode imputation


The main function responsible for the functionality is found on automl.py and is called AutoML.

Input: 
1. the features with missing values, 
2. the outcome variable.
3. the time in minutes.

Prints: 
1. Prints the accuracy performance.
2. Prints the best configuration.

Output: 
1. The imputed dataset with outcome variable included.


Conclusions from the paper:
1. Mean-Mode Imputation is performing equally or better than more sophisticated methods.
2. In real-world datasets, Denoise Auto-Encoders outperform Mean-Mode on some datasets, but the difference on average is not significant.
3. In simulated missing data, MissForest is on average better than the other methods, closely followed by Mean-Mode.


