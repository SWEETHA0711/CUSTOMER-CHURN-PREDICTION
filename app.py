import streamlit as st
import pandas as pd
import numpy as np
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
st.markdown("End-to-end Machine Learning Project using Random Forest")

# ===================== LOAD DATA =====================
@st.cache_data
def load_data():
    return pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")

df = load_data()

st.header("1️⃣ Dataset Overview")
st.write(df.head())
st.write("Dataset Shape:", df.shape)

# ===================== EDA =====================
st.header("2️⃣ Exploratory Data Analysis")

fig, ax = plt.subplots()
df['Churn'].value_counts().plot(kind='bar', ax=ax)
ax.set_title("Churn Distribution")
st.pyplot(fig)

fig, ax = plt.subplots()
df.boxplot(column='tenure', by='Churn', ax=ax)
ax.set_title("Churn vs Tenure")
plt.suptitle("")
st.pyplot(fig)

fig, ax = plt.subplots()
df.boxplot(column='MonthlyCharges', by='Churn', ax=ax)
ax.set_title("Churn vs Monthly Charges")
plt.suptitle("")
st.pyplot(fig)

# ===================== DATA CLEANING =====================
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
df.drop_duplicates(inplace=True)

# ===================== FEATURE ENGINEERING =====================
df['AvgChargesPerMonth'] = df['TotalCharges'] / (df['tenure'] + 1)
df['IsLongTermContract'] = df['Contract'].apply(lambda x: 1 if x in ['One year','Two year'] else 0)

service_features = internet_cols
df['NumServicesSubscribed'] = df[service_features].apply(lambda x: sum(x == 'Yes'), axis=1)

df['HighMonthlyCharge'] = (df['MonthlyCharges'] > df['MonthlyCharges'].quantile(0.75)).astype(int)
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

# ===================== CLASS BALANCE =====================
st.header("3️⃣ Class Balance")
fig, ax = plt.subplots()
y_train.value_counts().plot(kind='pie', autopct='%1.1f%%', ax=ax)
ax.set_ylabel("")
st.pyplot(fig)

# ===================== MODEL & TUNING =====================
st.header("4️⃣ Model Training & Hyperparameter Tuning")

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

st.write("Best Parameters:", grid.best_params_)

# ===================== EVALUATION =====================
st.header("5️⃣ Model Evaluation")

y_pred = rf_best.predict(X_test)
y_prob = rf_best.predict_proba(X_test)[:,1]

st.metric("Accuracy", round(accuracy_score(y_test, y_pred), 3))
st.metric("Precision", round(precision_score(y_test, y_pred), 3))
st.metric("Recall", round(recall_score(y_test, y_pred), 3))
st.metric("F1 Score", round(f1_score(y_test, y_pred), 3))
st.metric("ROC AUC", round(roc_auc_score(y_test, y_prob), 3))

# Confusion Matrix
fig, ax = plt.subplots()
ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred)).plot(ax=ax)
st.pyplot(fig)

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_prob)
fig, ax = plt.subplots()
ax.plot(fpr, tpr, label="ROC Curve")
ax.plot([0,1],[0,1],'--')
ax.legend()
st.pyplot(fig)

# ===================== FEATURE IMPORTANCE =====================
st.header("6️⃣ Feature Importance")

importances = pd.Series(rf_best.feature_importances_, index=X_encoded.columns)
top_features = importances.sort_values(ascending=False).head(10)

fig, ax = plt.subplots()
top_features.plot(kind='barh', ax=ax)
ax.invert_yaxis()
st.pyplot(fig)

# ===================== PDP =====================
st.header("7️⃣ Partial Dependence Plots")

top_3 = top_features.index[:3]
fig, ax = plt.subplots(figsize=(10,4))
PartialDependenceDisplay.from_estimator(rf_best, X_train, features=top_3, ax=ax)
st.pyplot(fig)

st.success("✅ Application Loaded Successfully")
