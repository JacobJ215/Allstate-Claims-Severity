# Allstate-Claims-Severity
## Allstate Claims Severity Kaggle Competition


This repository contains code for my submission to the Allstate Claims Severity Kaggle competition. The goal of the competition is to predict the loss value for an insurance claim, given a set of anonymized features.

### Code
The code is written in Python, using various libraries including pandas, numpy, matplotlib, seaborn, scipy, sklearn, xgboost, and lightgbm. The code can be found in the allstate_claims_severity.ipynb Jupyter notebook.

### Data
The train and test datasets are provided by Kaggle and can be downloaded from the competition website here. The datasets contain anonymized features and the corresponding loss value for each insurance claim.

### Approach
The code contains the following sections:

1. Import Libraries
2.  Load Data
3. Distribution of Target Variable
4. Data Preprocessing
5. Model Experimentation
6. Hyperparameter Tuning

In the data preprocessing step, the following steps are taken:

1. Drop loss values that exceed 20000
2. Separate the target variable from the features in the train dataset
3. Drop the id from the test dataset
4. Log transform target variable
5. Define categorical and continuous features
6. Define the preprocessing steps for categorical and continuous features
7. Preprocess train data
8. Split the train dataset into train and validation sets
9. Preprocess test data

Two base models, XGBoost and LightGBM, are trained on the preprocessed data, and their performance is evaluated using the mean absolute error (MAE) on the validation set.

In the hyperparameter tuning step, a random search with cross-validation is performed to find the best hyperparameters for each base model. The final MAE is calculated on the validation set using the best model for each base model.

### Results
The final MAE on the validation set is 0.4162073003058776 for XGBoost and 0.4156538377657095 for LightGBM.

LightGBM was selected as the final model for which we used it to make our predictions on the test dataset and submit our results for the competition.

