# Day 17 Report — Hyperparameter Tuning

## Technical Summary

Implemented GridSearchCV to automatically find the optimal hyperparameters for a Random Forest model.

## Implementation

* Defined parameter grid for estimators, depth, and split criteria
* Ran GridSearch with 3-Fold Cross Validation
* Compared baseline vs tuned model performance
* Calculated total number of models trained
* Saved best model using joblib

## Key Insight

The tuned model achieved higher accuracy than the baseline model, proving that hyperparameter optimization improves performance.

## Optimization Logic

GridSearch improves accuracy but increases training time significantly. For large datasets, RandomizedSearchCV is preferred to reduce computation.

## Reflection

GridSearch tests all combinations, which becomes computationally expensive as parameters increase. RandomizedSearchCV provides a faster alternative by sampling parameter combinations.
