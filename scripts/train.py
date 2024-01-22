import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from scipy import stats

BASE_DIR = "../data"
TRAIN_CSV = f"{BASE_DIR}/train.csv"

df_train = pd.read_csv(TRAIN_CSV)


def preprocess(df):
    DROP_COLUMNS = ['id', 'CustomerId', 'Surname']
    df = df.drop(DROP_COLUMNS, axis=1)
    
    ONE_HOT_ENCODE_COLUMNS = ['Geography', 'Gender']
    df = pd.get_dummies(df, columns=ONE_HOT_ENCODE_COLUMNS, drop_first=True)
    
    NUMERICAL_COLUMNS = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'EstimatedSalary']
    scaler = StandardScaler()
    df[NUMERICAL_COLUMNS] = scaler.fit_transform(df[NUMERICAL_COLUMNS])
        
    return df

df_train = preprocess(df_train)

X = df_train.drop('Exited', axis=1)
y = df_train['Exited']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a Random Forest Classifier
rf_classifier = RandomForestClassifier(
    n_estimators=150,
    max_depth=15,
    min_samples_split=2,
    min_samples_leaf=1,
    max_features=None,
    bootstrap=True,
    random_state=42
)

# Fit the model on the training data
rf_classifier.fit(X_train, y_train)

# Save the trained model to a file
joblib.dump(rf_classifier, '../models/rf_model.joblib')
