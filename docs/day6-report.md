# Day 6 Report — AI/ML Developer Track

## Technical Summary

Today I learned Logistic Regression for binary classification.
I trained a model to predict pass/fail outcomes based on hours of sleep and coffee consumption.
I also learned how to evaluate model performance using a confusion matrix and classification metrics.

## Bug Log

Issue: UserWarning about missing feature names during manual prediction
Cause: Manual test input was passed as a plain list instead of a labeled DataFrame
Solution: Converted manual input into a Pandas DataFrame with matching feature names to ensure consistency

## Conceptual Reflection

In healthcare applications like cancer detection, False Negatives are more dangerous than False Positives.
A False Negative can miss a serious disease, delaying treatment and increasing health risks.
