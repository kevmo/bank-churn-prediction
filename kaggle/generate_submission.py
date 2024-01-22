import pandas as pd
import requests

# Load the test data
test_data = pd.read_csv('../data/test.csv')  # Update the path accordingly

# Flask app URL
app_url = 'http://localhost:5000/predict'  # Update with your Flask app's URL

# List to store predictions
predictions_list = []

# Iterate through each row in the test data
for index, row in test_data.iterrows():
    # Extract the features (excluding the target variable)
    features = row.drop(['Exited']).to_dict()

    # Make a POST request to the Flask app for predictions
    response = requests.post(app_url, json=features)

    # Extract predictions from the response
    predictions = response.json()

    # Store the results (you might need to adjust this based on your actual output format)
    result = {
        'id': row['CustomerId'],
        'Exited': predictions[0]  # Assuming the prediction is a single value
    }

    predictions_list.append(result)

# Create a DataFrame from the predictions list
predictions_df = pd.DataFrame(predictions_list)

# Save the predictions to a CSV file
predictions_df.to_csv('predictions.csv', index=False)
