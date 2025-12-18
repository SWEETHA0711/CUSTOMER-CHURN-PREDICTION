import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    ConfusionMatrixDisplay, roc_curve
)
from sklearn.inspection import PartialDependenceDisplay

st.set_page_config(page_title="Customer Churn Prediction", layout="wide")

st.title("📊 Customer Churn Prediction System")

# ================= DATA LOAD =================
@st.cache_data
def load_data():
    return pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")

df = load_data()

# ================= DATASET PREVIEW =================
st.subheader("Dataset Overview")
st.dataframe(df.head())

# ================= CLASS BALANCE =================
st.subheader("Churn Class Distribution")

fig, ax = plt.subplots()
df['Churn'].value_counts().plot(kind='bar', ax=ax)
ax.set_xlabel("Churn")
ax.set_ylabel("Count")
st.pyplot(fig)

# ================= DATA CLEANING =================
df['MultipleLines'] = df['MultipleLines'].replace('No phone service', 'No')

internet_cols = [
    'OnlineSecurity','OnlineBackup','DeviceProtection',
    'TechSupport','StreamingTV','StreamingMovies'
]
for col in internet_cols:
    df[col] = df[col].replace('No internet service', 'No')

df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)
df.drop('customerID', axis=1, inplace=True)

# ================= FEATURE ENGINEERING =================
df['AvgChargesPerMonth'] = df['TotalCharges'] / (df['tenure'] + 1)
df['IsLongTermContract'] = df['Contract'].apply(lambda x: 1 if x in ['One year','Two year'] else 0)

services = [
    'OnlineSecurity','OnlineBackup','DeviceProtection',
    'TechSupport','StreamingTV','StreamingMovies'
]
df['NumServicesSubscribed'] = df[services].apply(lambda x: sum(x=='Yes'), axis=1)
df['HighMonthlyCharge'] = (df['MonthlyCharges'] > df['MonthlyCharges'].quantile(0.75)).astype(int)
df['SeniorCitizen'] = df['SeniorCitizen'].astype(int)

# ================= MODEL PREPARATION =================
X = df.drop('Churn', axis=1)
y = df['Churn'].map({'No': 0, 'Yes': 1})

X = pd.get_dummies(X, drop_first=True)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# ================= MODEL TRAINING =================
rf = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight='balanced'
)
rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)
y_prob = rf.predict_proba(X_test)[:, 1]

# ================= METRICS =================
st.subheader("Model Performance Metrics")

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Accuracy", round(accuracy_score(y_test, y_pred), 3))
col2.metric("Precision", round(precision_score(y_test, y_pred), 3))
col3.metric("Recall", round(recall_score(y_test, y_pred), 3))
col4.metric("F1 Score", round(f1_score(y_test, y_pred), 3))
col5.metric("ROC AUC", round(roc_auc_score(y_test, y_prob), 3))

# ================= CONFUSION MATRIX =================
st.subheader("Confusion Matrix")

cm = confusion_matrix(y_test, y_pred)
fig_cm, ax_cm = plt.subplots()
ConfusionMatrixDisplay(cm).plot(ax=ax_cm)
st.pyplot(fig_cm)

# ================= ROC CURVE =================
st.subheader("ROC Curve")

fpr, tpr, _ = roc_curve(y_test, y_prob)
fig_roc, ax_roc = plt.subplots()
ax_roc.plot(fpr, tpr, label="ROC Curve")
ax_roc.plot([0,1], [0,1], linestyle='--')
ax_roc.set_xlabel("False Positive Rate")
ax_roc.set_ylabel("True Positive Rate")
ax_roc.legend()
st.pyplot(fig_roc)

# ================= FEATURE IMPORTANCE =================
st.subheader("Feature Importance")

importances = rf.feature_importances_
indices = np.argsort(importances)[-15:]

fig_fi, ax_fi = plt.subplots()
ax_fi.barh(range(len(indices)), importances[indices])
ax_fi.set_yticks(range(len(indices)))
ax_fi.set_yticklabels(np.array(X.columns)[indices])
st.pyplot(fig_fi)

# ================= PARTIAL DEPENDENCE =================
st.subheader("Partial Dependence Plots")

top_features = np.array(X.columns)[indices[-3:]]

fig_pdp, ax_pdp = plt.subplots(figsize=(12,4))
PartialDependenceDisplay.from_estimator(
    rf, X_train, features=top_features, ax=ax_pdp
)
st.pyplot(fig_pdp)

st.success("✅ All visualizations generated successfully")
