from sklearn.preprocessing import StandardScaler
import pandas as pd


def preprocess(df):
    DROP_COLUMNS = ['id', 'CustomerId', 'Surname']
    df = df.drop(DROP_COLUMNS, axis=1)
    
    ONE_HOT_ENCODE_COLUMNS = ['Geography', 'Gender']

    df = pd.get_dummies(df, columns=ONE_HOT_ENCODE_COLUMNS, drop_first=True)
    
    NUMERICAL_COLUMNS = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'EstimatedSalary']

    scaler = StandardScaler()
    df[NUMERICAL_COLUMNS] = scaler.fit_transform(df[NUMERICAL_COLUMNS])
    
    print(df.columns)
    
    return df