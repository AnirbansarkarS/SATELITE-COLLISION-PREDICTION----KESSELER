import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

os.makedirs("outputs/results", exist_ok=True)

def train_logistic_model(df, feature_cols, target_col, return_preds=False):
    X = df[feature_cols]
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)

    print("✅ Model trained successfully")
    print(f"Accuracy: {acc:.3f}")
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))

    # Save model
    joblib.dump(model, "outputs/results/logistic_model.pkl")

    if return_preds:
        preds_df = pd.DataFrame({"y_test": y_test, "y_pred": y_pred})
        preds_df.to_csv("outputs/results/logistic_predictions.csv", index=False)
        return model, acc, y_test, y_pred

    return model, acc
