import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
import joblib
import os
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image

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

# Load model & encoders
@st.cache_resource
def load_model_and_encoders(model_dir='model_outputs'):
    try:
        model = joblib.load(os.path.join(model_dir, 'classification_model.pkl'))
        status_encoder = joblib.load(os.path.join(model_dir, 'status_encoder.pkl'))
        dayofweek_encoder = joblib.load(os.path.join(model_dir, 'dayofweek_encoder.pkl'))
        classification_tag_encoder = joblib.load(os.path.join(model_dir, 'classification_tag_encoder.pkl'))
        return model, status_encoder, dayofweek_encoder, classification_tag_encoder
    except FileNotFoundError as e:
        st.error(f"Could not load model or encoders: {e}")
        return None, None, None, None

# Load dataset
@st.cache_data
def load_data(file_path='data_processing/modified_dataset.parquet'):
    try:
        return pd.read_parquet(file_path)
    except FileNotFoundError:
        st.error(f"Could not load data from {file_path}")
        return None

# Function to create confusion matrix visualization
def plot_confusion_matrix(y_true, y_pred, class_names):
    cm = confusion_matrix(y_true, y_pred)
    fig = px.imshow(cm, labels=dict(x="Predicted", y="True", color="Count"), x=class_names, y=class_names, text_auto=True, color_continuous_scale='Blues')
    return fig

# Function to plot feature importance
def plot_feature_importance(model, feature_names):
    feature_importance = model.feature_importances_
    sorted_idx = np.argsort(feature_importance)
    fig = px.bar(x=feature_importance[sorted_idx], y=[feature_names[i] for i in sorted_idx], orientation='h', color=feature_importance[sorted_idx], title="Feature Importance")
    return fig

# Function to preprocess data
def preprocess_features(df, status_encoder, dayofweek_encoder):
    features = df.drop('Classification_Tag', axis=1).copy()
    if 'Status' in features.columns:
        features['Status'] = features['Status'].map({name: i for i, name in enumerate(status_encoder.classes_)})
    if 'DayOfWeek' in features.columns:
        features['DayOfWeek'] = features['DayOfWeek'].map({name: i for i, name in enumerate(dayofweek_encoder.classes_)})
    features = features.apply(pd.to_numeric, errors='coerce').fillna(features.mean())
    return features

# Main layout
st.markdown('<div class="main-header">Enhanced ML Dashboard</div>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("https://xgboost.ai/images/logo/xgboost-logo.png", width=200)
    model_dir = st.text_input("Model Directory", "model_outputs")
    data_path = st.text_input("Data Path", "data_processing/modified_dataset.parquet")
    if st.button("Load Model and Data", key="load_button"):
        st.session_state['load_triggered'] = True

if 'load_triggered' not in st.session_state:
    st.session_state['load_triggered'] = False

model, status_encoder, dayofweek_encoder, classification_tag_encoder = load_model_and_encoders(model_dir)
df = load_data(data_path)

if not st.session_state['load_triggered']:
    st.info("Please load the model and data using the sidebar button.")
    st.stop()

if model is None or df is None:
    st.error("Could not proceed without model or data. Check paths.")
    st.stop()

features = preprocess_features(df, status_encoder, dayofweek_encoder)
scaler = StandardScaler()
scaler.fit(features)
class_names = classification_tag_encoder.classes_

# Tabs for navigation
tabs = st.tabs(["Model Performance", "Feature Analysis", "Prediction Tool", "Dataset Exploration"])

# Model Performance
with tabs[0]:
    st.subheader("Model Performance")
    y_true, y_pred = df['Classification_Tag'], model.predict(features)
    st.text(classification_report(y_true, y_pred))
    st.plotly_chart(plot_confusion_matrix(y_true, y_pred, class_names))

# Feature Importance
with tabs[1]:
    st.subheader("Feature Importance")
    st.plotly_chart(plot_feature_importance(model, features.columns))

# Prediction Tool
with tabs[2]:
    st.subheader("Make a Prediction")
    input_data = {col: st.number_input(f"{col}", float(df[col].min()), float(df[col].max()), float(df[col].mean())) for col in df.columns if col != 'Classification_Tag'}
    if st.button("Predict"):
        input_scaled = scaler.transform(pd.DataFrame([input_data]))
        prediction = model.predict(input_scaled)
        st.success(f"Predicted Class: {classification_tag_encoder.inverse_transform(prediction)[0]}")

# Dataset Exploration
with tabs[3]:
    st.subheader("Dataset Overview")
    st.dataframe(df.sample(5))
    st.plotly_chart(px.histogram(df, x=df.columns[0], nbins=30, title=f"Distribution of {df.columns[0]}"))
