import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score
import matplotlib.pyplot as plt

st.set_page_config(page_title="Customer Churn Prediction", layout="wide")

st.title("📊 Customer Churn Prediction System")
st.write("This application predicts customer churn using a Random Forest model.")

# ================= DATA LOADING =================
@st.cache_data
def load_data():
    df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")
    return df

df = load_data()

st.subheader("Dataset Preview")
st.dataframe(df.head())

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
model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight='balanced'
)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:,1]

# ================= METRICS =================
st.subheader("Model Performance Metrics")

col1, col2, col3, col4, col5 = st.columns(5)

col1.metric("Accuracy", round(accuracy_score(y_test, y_pred), 3))
col2.metric("Precision", round(precision_score(y_test, y_pred), 3))
col3.metric("Recall", round(recall_score(y_test, y_pred), 3))
col4.metric("F1 Score", round(f1_score(y_test, y_pred), 3))
col5.metric("ROC AUC", round(roc_auc_score(y_test, y_prob), 3))

# ================= CHURN DISTRIBUTION =================
st.subheader("Churn Distribution")

fig, ax = plt.subplots()
y.value_counts().plot(kind='bar', ax=ax)
ax.set_xlabel("Churn")
ax.set_ylabel("Count")
st.pyplot(fig)

st.success("✅ Model executed successfully!")
