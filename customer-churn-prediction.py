import os
import sys
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import messagebox

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    ConfusionMatrixDisplay, roc_curve
)

# ===================== RESOURCE PATH =====================
def resource_path(relative_path):
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

# ===================== MAIN FUNCTION =====================
def main():

    # Create output folder
    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)

    # ===================== LOAD DATA =====================
    csv_path = resource_path("WA_Fn-UseC_-Telco-Customer-Churn.csv")
    df = pd.read_csv(csv_path)

    # ===================== DATA CLEANING =====================
    df['MultipleLines'] = df['MultipleLines'].replace('No phone service', 'No')

    internet_cols = [
        'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
        'TechSupport', 'StreamingTV', 'StreamingMovies'
    ]
    for col in internet_cols:
        df[col] = df[col].replace('No internet service', 'No')

    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)
    df.drop('customerID', axis=1, inplace=True)
    df.drop_duplicates(inplace=True)

    # ===================== FEATURE ENGINEERING =====================
    df['AvgChargesPerMonth'] = df['TotalCharges'] / (df['tenure'] + 1)
    df['IsLongTermContract'] = df['Contract'].apply(
        lambda x: 1 if x in ['One year', 'Two year'] else 0
    )

    service_features = [
        'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
        'TechSupport', 'StreamingTV', 'StreamingMovies'
    ]
    df['NumServicesSubscribed'] = df[service_features].apply(
        lambda row: sum(row == 'Yes'), axis=1
    )

    df['HighMonthlyCharge'] = (
        df['MonthlyCharges'] > df['MonthlyCharges'].quantile(0.75)
    ).astype(int)

    df['SeniorCitizen'] = df['SeniorCitizen'].astype(int)

    # ===================== PREPARATION =====================
    X = df.drop('Churn', axis=1)
    y = df['Churn'].map({'No': 0, 'Yes': 1})

    categorical_cols = X.select_dtypes(include='object').columns
    numerical_cols = X.select_dtypes(exclude='object').columns

    X_encoded = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

    scaler = StandardScaler()
    X_encoded[numerical_cols] = scaler.fit_transform(X_encoded[numerical_cols])

    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y, test_size=0.2, random_state=42, stratify=y
    )

    # ===================== MODEL TRAINING =====================
    rf = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight='balanced'
    )
    rf.fit(X_train, y_train)

    # ===================== HYPERPARAMETER TUNING =====================
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [None, 20],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2],
        'class_weight': ['balanced']
    }

    grid = GridSearchCV(
        RandomForestClassifier(random_state=42),
        param_grid,
        scoring='recall',
        cv=3,
        n_jobs=-1
    )
    grid.fit(X_train, y_train)

    rf_best = grid.best_estimator_

    # ===================== EVALUATION =====================
    y_pred = rf_best.predict(X_test)
    y_prob = rf_best.predict_proba(X_test)[:, 1]

    metrics_text = f"""
Accuracy  : {accuracy_score(y_test, y_pred):.4f}
Precision : {precision_score(y_test, y_pred):.4f}
Recall    : {recall_score(y_test, y_pred):.4f}
F1 Score  : {f1_score(y_test, y_pred):.4f}
ROC AUC   : {roc_auc_score(y_test, y_prob):.4f}
"""

    # ===================== CONFUSION MATRIX =====================
    cm = confusion_matrix(y_test, y_pred)
    ConfusionMatrixDisplay(cm).plot()
    plt.title("Confusion Matrix")
    plt.savefig(os.path.join(output_dir, "confusion_matrix.png"))
    plt.show()

    # ===================== ROC CURVE =====================
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plt.plot(fpr, tpr, label="ROC Curve")
    plt.plot([0, 1], [0, 1], '--')
    plt.legend()
    plt.title("ROC Curve")
    plt.savefig(os.path.join(output_dir, "roc_curve.png"))
    plt.show()

    # ===================== FINAL POPUP =====================
    root = tk.Tk()
    root.withdraw()
    messagebox.showinfo(
        "Customer Churn Prediction",
        "Execution completed successfully!\n\nResults saved in 'outputs' folder.\n\n" + metrics_text
    )
    root.destroy()

# ===================== RUN =====================
if __name__ == "__main__":
    main()
