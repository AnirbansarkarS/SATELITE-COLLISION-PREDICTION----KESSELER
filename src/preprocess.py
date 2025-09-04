import pandas as pd
from sklearn.preprocessing import LabelEncoder

def preprocess_data(df):
    """
    Preprocess the dataset:
    - Encode categorical features
    - Handle timestamps
    """

    # Convert timestamp to datetime and extract useful features
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])
    df["Hour"] = df["Timestamp"].dt.hour
    df["Day"] = df["Timestamp"].dt.day
    df["Month"] = df["Timestamp"].dt.month

    # Encode categorical variables
    categorical_cols = ["Location", "Object_Type", "Peak_Time"]
    encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le

    return df, encoders
