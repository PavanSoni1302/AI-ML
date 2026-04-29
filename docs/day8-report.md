# Day 8 Report — AI/ML Developer Track

## Technical Summary

Today I built a complete end-to-end machine learning pipeline using the California Housing dataset.
I performed data loading, preprocessing, scaling, model training, prediction, and evaluation.

## Bug Log

Issue: NameError for y_test and predictions in Jupyter Notebook
Cause: Variables were not defined in the current notebook cell before use
Solution: Executed the full pipeline in a single cell to ensure all variables were defined before plotting residuals

## Conceptual Reflection

If the residual plot shows a U-shape, it indicates that the linear model is not capturing the relationship correctly and that the data likely has a non-linear pattern.
