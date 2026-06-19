from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load model
model = joblib.load('production_model.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    # Convert JSON → DataFrame
    df = pd.DataFrame([data])

    # Predict
    prediction = model.predict(df)

    return jsonify({
        "prediction": prediction.tolist()
    })

if __name__ == '__main__':
    app.run(port=5000, debug=True)