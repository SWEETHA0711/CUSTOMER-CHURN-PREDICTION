import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    ConfusionMatrixDisplay, roc_curve
)
from sklearn.inspection import PartialDependenceDisplay

st.set_page_config(page_title="Customer Churn Prediction", layout="wide")
st.title("📊 Customer Churn Prediction System")
st.markdown("Random Forest based Customer Churn Analysis")

# ===================== LOAD ARTIFACTS =====================
@st.cache_resource
def load_artifacts():
    model = joblib.load("rf_model.pkl")
    scaler = joblib.load("scaler.pkl")
    features = joblib.load("features.pkl")
    return model, scaler, features

rf_model, scaler, feature_cols = load_artifacts()

# ===================== LOAD DATA =====================
@st.cache_data
def load_data():
    return pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")

df = load_data()

st.header("1️⃣ Dataset Overview")
st.write(df.head())
st.write("Dataset Shape:", df.shape)

# ===================== DATA CLEANING =====================
df['MultipleLines'] = df['MultipleLines'].replace('No phone service', 'No')

internet_cols = [
    'OnlineSecurity','OnlineBackup','DeviceProtection',
    'TechSupport','StreamingTV','StreamingMovies'
]
for col in internet_cols:
    df[col] = df[col].replace('No internet service', 'No')

df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())
df.drop('customerID', axis=1, inplace=True)
df.drop_duplicates(inplace=True)

# ===================== FEATURE ENGINEERING =====================
df['AvgChargesPerMonth'] = df['TotalCharges'] / (df['tenure'] + 1)
df['IsLongTermContract'] = df['Contract'].apply(lambda x: 1 if x in ['One year','Two year'] else 0)

df['NumServicesSubscribed'] = df[internet_cols].apply(lambda x: sum(x == 'Yes'), axis=1)
df['HighMonthlyCharge'] = (df['MonthlyCharges'] > df['MonthlyCharges'].quantile(0.75)).astype(int)
df['SeniorCitizen'] = df['SeniorCitizen'].astype(int)

# ===================== PREPARATION =====================
X = df.drop('Churn', axis=1)
y = df['Churn'].map({'No': 0, 'Yes': 1})

X_encoded = pd.get_dummies(X, drop_first=True)
X_encoded = X_encoded.reindex(columns=feature_cols, fill_value=0)
X_encoded[X_encoded.columns] = scaler.transform(X_encoded)

# ===================== PREDICTIONS =====================
y_pred = rf_model.predict(X_encoded)
y_prob = rf_model.predict_proba(X_encoded)[:,1]

# ===================== CLASS BALANCE =====================
st.header("2️⃣ Class Balance")

fig, ax = plt.subplots()
y.value_counts().plot(kind='pie', autopct='%1.1f%%', ax=ax)
ax.set_ylabel("")
st.pyplot(fig)

# ===================== MODEL METRICS =====================
st.header("3️⃣ Model Performance")

st.metric("Accuracy", round(accuracy_score(y, y_pred), 3))
st.metric("Precision", round(precision_score(y, y_pred), 3))
st.metric("Recall", round(recall_score(y, y_pred), 3))
st.metric("F1 Score", round(f1_score(y, y_pred), 3))
st.metric("ROC AUC", round(roc_auc_score(y, y_prob), 3))

# ===================== CONFUSION MATRIX =====================
fig, ax = plt.subplots()
ConfusionMatrixDisplay(confusion_matrix(y, y_pred)).plot(ax=ax)
st.pyplot(fig)

# ===================== ROC CURVE =====================
fpr, tpr, _ = roc_curve(y, y_prob)
fig, ax = plt.subplots()
ax.plot(fpr, tpr)
ax.plot([0,1],[0,1],'--')
ax.set_title("ROC Curve")
st.pyplot(fig)

# ===================== FEATURE IMPORTANCE =====================
st.header("4️⃣ Feature Importance")

importances = pd.Series(rf_model.feature_importances_, index=feature_cols)
top_features = importances.sort_values(ascending=False).head(10)

fig, ax = plt.subplots()
top_features.plot(kind='barh', ax=ax)
ax.invert_yaxis()
st.pyplot(fig)

# ===================== PDP =====================
st.header("5️⃣ Partial Dependence Plots")

top_3 = top_features.index[:3]
fig, ax = plt.subplots(figsize=(10,4))
PartialDependenceDisplay.from_estimator(rf_model, X_encoded, features=top_3, ax=ax)
st.pyplot(fig)

# ===================== CORRELATION HEATMAP =====================
st.header("6️⃣ Correlation Heatmap")

numeric_features = [
    'tenure','MonthlyCharges','TotalCharges',
    'AvgChargesPerMonth','NumServicesSubscribed',
    'HighMonthlyCharge','IsLongTermContract','SeniorCitizen'
]

corr = X_encoded[numeric_features].corr()
fig, ax = plt.subplots(figsize=(8,6))
im = ax.imshow(corr, cmap='coolwarm')
ax.set_xticks(range(len(numeric_features)))
ax.set_yticks(range(len(numeric_features)))
ax.set_xticklabels(numeric_features, rotation=45, ha="right")
ax.set_yticklabels(numeric_features)
fig.colorbar(im)
st.pyplot(fig)

# ===================== PROBABILITY HISTOGRAM =====================
st.header("7️⃣ Predicted Probability Distribution")

fig, ax = plt.subplots(figsize=(8,5))
ax.hist(y_prob[y==0], bins=20, alpha=0.7, label='No Churn')
ax.hist(y_prob[y==1], bins=20, alpha=0.7, label='Churn')
ax.legend()
st.pyplot(fig)

# ===================== PDP + ICE =====================
st.header("8️⃣ PDP & ICE Plots")

top_numeric = ['tenure','MonthlyCharges','NumServicesSubscribed']
fig, ax = plt.subplots(figsize=(12,6))
PartialDependenceDisplay.from_estimator(
    rf_model,
    X_encoded,
    features=top_numeric,
    kind='both',
    ax=ax
)
st.pyplot(fig)

st.success("✅ Application Loaded Successfully")
