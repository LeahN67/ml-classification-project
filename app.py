import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import os
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

# Set page configuration
st.set_page_config(page_title="Enhanced ML Dashboard", page_icon="📊", layout="wide")

# Custom styling
st.markdown("""
    <style>
        .main-header {text-align: center; font-size: 2.5rem; color: #1E88E5; font-weight: bold;}
        .card {background-color: #f8f9fa; border-radius: 10px; padding: 15px; margin: 10px;}
        .metric-card {background-color: #e3f2fd; border-radius: 10px; padding: 10px; text-align: center;}
    </style>
""", unsafe_allow_html=True)

# Load model & data
@st.cache_resource
def load_model(model_path='model_outputs/classification_model.pkl'):
    return joblib.load(model_path)

@st.cache_data
def load_data(data_path='data_processing/modified_dataset.parquet'):
    return pd.read_parquet(data_path)

# Function to generate a confusion matrix plot
def plot_confusion_matrix(y_true, y_pred, class_names):
    cm = confusion_matrix(y_true, y_pred)
    fig = px.imshow(cm, labels=dict(x="Predicted", y="True", color="Count"), x=class_names, y=class_names, text_auto=True, color_continuous_scale='Blues')
    return fig

# Function to generate feature importance plot
def plot_feature_importance(model, feature_names):
    feature_importance = model.feature_importances_
    sorted_idx = np.argsort(feature_importance)
    fig = px.bar(x=feature_importance[sorted_idx], y=[feature_names[i] for i in sorted_idx], orientation='h', color=feature_importance[sorted_idx], title="Feature Importance")
    return fig

# Main dashboard layout
st.markdown('<div class="main-header">Enhanced ML Dashboard</div>', unsafe_allow_html=True)

# Sidebar options
with st.sidebar:
    st.image("https://xgboost.ai/images/logo/xgboost-logo.png", width=200)
    model_path = st.text_input("Model Path", "model_outputs/classification_model.pkl")
    data_path = st.text_input("Data Path", "data_processing/modified_dataset.parquet")
    load_button = st.button("Load Model & Data")

# Load data & model
if load_button:
    model = load_model(model_path)
    df = load_data(data_path)
    st.success("Model & Data Loaded Successfully!")
else:
    st.warning("Click 'Load Model & Data' to proceed.")
    st.stop()

# Tabs for navigation
tabs = st.tabs(["Model Performance", "Feature Analysis", "Prediction Tool", "Dataset Exploration"])

# Tab 1: Model Performance
with tabs[0]:
    st.subheader("Model Performance")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Classification Report")
        y_true, y_pred = df['Classification_Tag'], model.predict(df.drop(columns=['Classification_Tag']))
        st.text(classification_report(y_true, y_pred))
    with col2:
        st.markdown("### Confusion Matrix")
        st.plotly_chart(plot_confusion_matrix(y_true, y_pred, df['Classification_Tag'].unique()))

# Tab 2: Feature Importance
with tabs[1]:
    st.subheader("Feature Importance")
    st.plotly_chart(plot_feature_importance(model, df.drop(columns=['Classification_Tag']).columns))

# Tab 3: Prediction Tool
with tabs[2]:
    st.subheader("Make a Prediction")
    input_data = {col: st.number_input(f"{col}", float(df[col].min()), float(df[col].max()), float(df[col].mean())) for col in df.columns if col != 'Classification_Tag'}
    if st.button("Predict"):
        prediction = model.predict(pd.DataFrame([input_data]))
        st.success(f"Predicted Class: {prediction[0]}")

# Tab 4: Dataset Exploration
with tabs[3]:
    st.subheader("Dataset Overview")
    st.dataframe(df.sample(5))
    st.plotly_chart(px.histogram(df, x=df.columns[0], nbins=30, title=f"Distribution of {df.columns[0]}"))
