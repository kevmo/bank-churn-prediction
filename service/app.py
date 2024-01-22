from flask import Flask, request, jsonify
from sklearn.externals import joblib
import pandas as pd

from scripts.preprocess import preprocess

app = Flask(__name__)

model = joblib.load('../models/rf_classifier.joblib')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        input_data = preprocess(pd.DataFrame(data))

        # Make predictions using the loaded model
        predictions = model.predict(input_data)

        # Return the predictions as JSON
        return jsonify(predictions.tolist())

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
