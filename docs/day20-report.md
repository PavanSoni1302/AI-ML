# Day 20 Report — End-to-End ML Pipeline & API Deployment

## Technical Summary

Today I transitioned from a standalone machine learning model to a production-ready system by building an end-to-end pipeline and deploying it as a REST API using Flask. This allows real-time predictions and integration with other services.

## Implementation

### 1. Model Serialization (Pipeline)

* Created a Pipeline combining:

  * StandardScaler (for preprocessing)
  * RandomForestRegressor (for prediction)
* Trained the pipeline on the California Housing dataset
* Saved the complete pipeline using Joblib as `production_model.pkl`

### 2. API Development

* Built a Flask-based REST API
* Created a `/predict` endpoint to accept JSON input
* Converted incoming JSON data into a DataFrame
* Used the saved pipeline to generate predictions
* Returned predictions in JSON format

### 3. API Testing (Integration Simulation)

* Created a separate script using the `requests` library
* Sent a POST request with sample user data
* Successfully received a prediction response from the API

## Key Insight

Using a Pipeline ensures that preprocessing and model prediction are applied consistently. It eliminates the risk of mismatched transformations and guarantees reliable predictions in production.

## Bug Log

**Issue 1:** ModuleNotFoundError for flask
**Cause:** Flask was not installed in the virtual environment
**Solution:** Installed Flask using pip inside the active `.venv`

**Issue 2:** Model file not found during API execution
**Cause:** API was run before generating the `.pkl` file
**Solution:** Ensured `train_and_save.py` was executed before running the API

**Issue 3:** API request failure due to missing input fields
**Cause:** JSON payload did not match model feature structure
**Solution:** Provided all required features in correct format

## Integration Understanding

This system is designed to integrate with other components:

* **Node.js Backend:** Sends user data to `/predict` endpoint and receives predictions
* **Data Analyst:** Can use API outputs for dashboards and analytics
* **DevOps Engineer:** Deploys and scales the API using containers or cloud infrastructure

This demonstrates how AI models function as a service within a larger system.

## Reflection

Saving a Pipeline object instead of separate scaler and model ensures that data preprocessing and prediction are always applied in the correct sequence. This reduces the risk of errors and simplifies deployment. It also makes the system more robust and easier to integrate with other services.

## Key Learning

Moving from a Jupyter-based model to a deployed API is essential for real-world applications. A model only becomes valuable when it can be accessed and used by other systems in real time.
