# Day 9 Report — AI/ML Developer Track

## Technical Summary

Today I learned hyperparameter tuning using GridSearchCV.
I optimized a Ridge Regression model by testing multiple alpha values and selecting the best one based on cross-validation.

## Bug Log

Issue: NameError for X_train_scaled in Jupyter Notebook
Cause: Variables were not defined in the current notebook session before using them
Solution: Executed the full pipeline (data loading, splitting, scaling) in a single cell before training the model

## Conceptual Reflection

Using a wide range of parameter values helps identify the best region quickly.
Starting with small increments may miss the optimal value entirely.
