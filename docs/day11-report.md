# Day 11 Report — AI/ML Developer Track

## Technical Summary

Today I implemented ensemble learning using Random Forest Regressor.
I combined multiple decision trees to improve model stability and accuracy.

## Bug Log

Issue: NameError for X_train and y_train in Jupyter Notebook
Cause: Train-test split was not executed before using the variables
Solution: Added dataset loading and train-test split before model training

## Conceptual Reflection

As the number of trees increases, accuracy improves initially but eventually reaches a point of diminishing returns where additional trees increase training time without significant improvement.
