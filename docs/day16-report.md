# Day 16 Report — Model Validation & K-Fold

## Technical Summary

Implemented K-Fold Cross Validation to evaluate model stability and eliminate dependency on a single train-test split.

## Implementation

* Applied 5-Fold Cross Validation using Random Forest
* Calculated Mean Accuracy and Standard Deviation
* Compared Training vs Validation Accuracy
* Performed Shuffle vs No-Shuffle experiment

## Key Insight

The model showed high accuracy with low standard deviation, indicating stability and good generalization.

## Overfitting Analysis

Training accuracy was slightly higher than validation accuracy, indicating no overfitting.

## Reflection

Shuffling ensures balanced data distribution across folds and prevents bias toward dominant groups, improving generalization.
