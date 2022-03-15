# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details

Maria Frik created the model. It is XgBoost using the default hyperparameters in xgboost 1.5.2 .

## Intended Use

This model should be used to predict whether the salary of an employee makes over 50K a year. It is based off a handful of attributes. 

## Training Data

The data was obtained from the UCI Machine Learning Repository (https://archive.ics.uci.edu/ml/datasets/census+income). Prediction task is to determine whether a person makes over 50K a year. 
Extraction was done by Barry Becker from the 1994 Census database. 

The original data set has 48842 rows, and a 70-30 split was used to break this into a train and test set. No stratification was done. To use the data for training a One Hot Encoder was used on the features and a label binarizer was used on the labels.

## Evaluation Data

For evaluation 30 percent of the original data was used.  Furthermore, metrics were evaluated on the data slices "workclass", "education", "marital-status", "occupation", "relationship", "race", "sex" and "native-country".

## Metrics

The model was evaluated using precision, recall and f beta score. The values are:
- precision: 0.7582
- recall: 0.6615
- fbeta: 0.7065

## Ethical Considerations

The model is not meant to be shown for social injustice in terms of salary such as in terms of sex, gender education. 

## Caveats and Recommendations

An optimisation in terms of hyperparameters or feature preprocessing steps is highly recommended.
