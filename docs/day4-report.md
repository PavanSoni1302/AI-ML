# Day 4 Report — AI/ML Developer Track

## ✅ Technical Summary

Today I worked on data pre-processing techniques, which is a critical part of the AI pipeline.
I learned how to handle missing data using Pandas (Data Imputation), scale features using MinMaxScaler from Scikit-Learn, and visualize relationships between variables using Seaborn heatmaps.

## 🐞 Bug Log

Issue: ModuleNotFoundError: No module named 'seaborn'
Cause: Seaborn library was not installed in the active virtual environment
Solution: Installed seaborn and matplotlib using pip inside the virtual environment and verified the installation before running the script again.

## 💡 Conceptual Reflection

Filling missing values with the mean is better than using 0 because 0 can distort the dataset and introduce bias.
Using the mean helps maintain the natural distribution of the data and allows the model to learn more accurate patterns.
