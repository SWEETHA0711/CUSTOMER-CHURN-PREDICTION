import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
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
st.markdown("End-to-End Machine Learning Project using Random Forest")

# ===================== LOAD DATA =====================
@st.cache_data
def load_data():
    return pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")

df = load_data()

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

# ===================== FEATURE ENGINEERING =====================
df['AvgChargesPerMonth'] = df['TotalCharges'] / (df['tenure'] + 1)
df['IsLongTermContract'] = df['Contract'].apply(lambda x: 1 if x in ['One year','Two year'] else 0)
df['NumServicesSubscribed'] = df[internet_cols].apply(lambda x: sum(x == 'Yes'), axis=1)
df['HighMonthlyCharge'] = (df['MonthlyCharges'] > df['MonthlyCharges'].quantile(0.75)).astype(int)
df['SeniorCitizen'] = df['SeniorCitizen'].astype(int)

# ===================== PREPARATION =====================
X = df.drop('Churn', axis=1)
y = df['Churn'].map({'No': 0, 'Yes': 1})

X = pd.get_dummies(X, drop_first=True)

scaler = StandardScaler()
X[X.columns] = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ===================== TRAIN MODEL (CACHED) =====================
@st.cache_resource
def train_model(X_train, y_train):
    model = RandomForestClassifier(
        n_estimators=150,
        class_weight='balanced',
        random_state=42
    )
    model.fit(X_train, y_train)
    return model

rf_model = train_model(X_train, y_train)

# ===================== DATASET OVERVIEW =====================
st.header("1️⃣ Dataset Overview")
st.write(df.head())
st.write("Dataset Shape:", df.shape)

# ===================== CLASS BALANCE =====================
st.header("2️⃣ Class Balance")
fig, ax = plt.subplots()
y_train.value_counts().plot(kind='pie', autopct='%1.1f%%', ax=ax)
ax.set_ylabel("")
st.pyplot(fig)

# ===================== MODEL EVALUATION =====================
st.header("3️⃣ Model Evaluation")

y_pred = rf_model.predict(X_test)
y_prob = rf_model.predict_proba(X_test)[:,1]

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
st.header("4️⃣ Feature Importance")

importances = pd.Series(rf_model.feature_importances_, index=X.columns)
top_features = importances.sort_values(ascending=False).head(10)

fig, ax = plt.subplots()
top_features.plot(kind='barh', ax=ax)
ax.invert_yaxis()
st.pyplot(fig)

# ===================== CORRELATION HEATMAP =====================
st.header("5️⃣ Correlation Heatmap")

numeric_features = [
    'tenure', 'MonthlyCharges', 'TotalCharges',
    'AvgChargesPerMonth', 'NumServicesSubscribed',
    'HighMonthlyCharge', 'IsLongTermContract', 'SeniorCitizen'
]

corr = X[numeric_features].corr()

fig, ax = plt.subplots(figsize=(8,6))
im = ax.imshow(corr, cmap='coolwarm')
ax.set_xticks(range(len(numeric_features)))
ax.set_yticks(range(len(numeric_features)))
ax.set_xticklabels(numeric_features, rotation=45, ha="right")
ax.set_yticklabels(numeric_features)
fig.colorbar(im)
st.pyplot(fig)

# ===================== PROBABILITY DISTRIBUTION =====================
st.header("6️⃣ Predicted Probability Distribution")

fig, ax = plt.subplots()
ax.hist(y_prob[y_test==0], bins=20, alpha=0.7, label='No Churn')
ax.hist(y_prob[y_test==1], bins=20, alpha=0.7, label='Churn')
ax.set_xlabel("Predicted Probability of Churn")
ax.legend()
st.pyplot(fig)

# ===================== PDP & ICE =====================
st.header("7️⃣ Partial Dependence & ICE Plots")

top_numeric = ['tenure', 'MonthlyCharges', 'NumServicesSubscribed']
fig, ax = plt.subplots(figsize=(12,6))
PartialDependenceDisplay.from_estimator(
    rf_model,
    X_train,
    features=top_numeric,
    kind='both',
    grid_resolution=50,
    ax=ax
)
st.pyplot(fig)

st.success("✅ Application Loaded Successfully")
