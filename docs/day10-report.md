# Day 10 Report — AI/ML Developer Track

## Technical Summary

Today I explored non-linear relationships using Polynomial Regression and Decision Trees.
I learned how polynomial features allow linear models to capture curved patterns, while decision trees use rule-based splits to model complex relationships.

I also analyzed how increasing model complexity (tree depth) affects performance and leads to overfitting.

## Bug Log

Issue: NameError for variables like X, y, and model in Jupyter Notebook
Cause: Variables were not defined in the current cell before use
Solution: Ensured that all required variables and models were defined within the same cell or executed previous cells before running dependent code

## Conceptual Reflection

A jittery model that closely follows every data point is not desirable because it indicates overfitting. Instead of learning the underlying pattern, the model memorizes noise present in the training data, leading to poor performance on unseen data. A smoother model captures the general trend and generalizes better for real-world predictions.
