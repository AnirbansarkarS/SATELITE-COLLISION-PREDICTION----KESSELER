# 🛰️ Satellite Collision Prediction using Logistic Regression

## 📌 Overview

This project predicts the probability of satellite collisions in orbit using **Logistic Regression**.
We leverage features such as **timestamp, orbital location, object type, traffic density, and peak-time activity** to model collision risks.

The goal is to build a **baseline interpretable model** for collision prediction before moving to more advanced ML methods.

---

## 📂 Project Structure

```
satellite-collision-predictor/
│
├── data/
│   ├── raw/                  # Original dataset (CSV)
│   ├── processed/            # Cleaned dataset after preprocessing
│
├── notebooks/
│   ├── 01-eda.ipynb          # Exploratory Data Analysis
│   └── 02-logistic-regression.ipynb  # Model training & evaluation
│
├── outputs/
│   ├── results/              # Saved models & evaluation metrics
│   └── figures/              # Plots (ROC curve, confusion matrix, etc.)
│
├── src/
│   ├── loader.py             # Data loading utilities
│   ├── preprocess.py         # Preprocessing (encoding, scaling, handling missing values)
│   ├── train.py              # Training logistic regression model
│   ├── evaluate.py           # Evaluation metrics (accuracy, precision, recall, ROC-AUC)
│   └── utils.py              # Helper functions (plots, logging, etc.)
│
├── requirements.txt          # Python dependencies
└── README.md                 # Project documentation
```

---

## ⚙️ Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/AirbansarkarS/.git
cd satellite-collision-predictor
pip install -r requirements.txt
```

---

## 📊 Dataset

* **Features:**

  * `Timestamp` → Time of observation
  * `Location` → Orbital location/zone
  * `Object_Type` → Type of object (satellite, debris, etc.)
  * `Traffic_Density` → Objects per orbital region
  * `Peak_Time` → Whether observation falls in a peak activity window

* **Target:**

  * `Collision` (0 = No, 1 = Yes)

Dataset should be placed in `data/raw/`.

---

## 🚀 Usage

### Preprocess the Data

```python
from src.loader import load_raw_csv, save_processed
from src.preprocess import preprocess_data

df = load_raw_csv("data/raw/satellite.csv")
df_processed = preprocess_data(df)
save_processed(df_processed, "data/processed/satellite_processed.csv")
```

### Train Logistic Regression

```python
from src.train import train_logistic_model
from src.evaluate import evaluate_model

model, X_test, y_test, y_pred = train_logistic_model(
    df_processed, feature_cols=["Location","Object_Type","Traffic_Density","Peak_Time"], target_col="Collision"
)

evaluate_model(y_test, y_pred)
```

---

## 📈 Evaluation Metrics

* **Accuracy**
* **Precision & Recall**
* **Confusion Matrix**
* **ROC-AUC Curve**

Results are stored under `outputs/results/`.

---

## 🔮 Future Improvements

* Include **orbital mechanics features** (velocity, altitude, inclination).
* Add **real-time traffic updates** from space-object catalogs (e.g., NORAD).
* Explore **tree-based methods (XGBoost, Random Forest)** for higher accuracy.

---

## 🤝 Contributing

Pull requests are welcome! If you’d like to add features (e.g., real satellite tracking APIs), please open an issue first.


