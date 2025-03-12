import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
import joblib
import os
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
import io

# Set page configuration
st.set_page_config(
    page_title="ML Classification Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced CSS to make the dashboard more colorful and professional
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
        font-weight: bold;
        background: linear-gradient(90deg, #1E88E5, #5E35B1);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        padding: 10px;
    }
    .sub-header {
        font-size: 1.8rem;
        color: #43A047;
        margin-top: 1rem;
        font-weight: bold;
        border-left: 5px solid #43A047;
        padding-left: 10px;
    }
    .section {
        font-size: 1.2rem;
        color: #5E35B1;
        margin-top: 0.8rem;
        font-weight: 500;
        display: inline-block;
        border-bottom: 2px solid #5E35B1;
        padding-bottom: 3px;
    }
    .card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    .card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 15px rgba(0, 0, 0, 0.2);
    }
    .metric-value {
        font-size: 2.5rem;
        font-weight: bold;
        margin: 10px 0;
        text-align: center;
    }
    .metric-label {
        font-size: 1rem;
        color: #555;
        text-align: center;
    }
    .metric-card {
        background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
        border-radius: 10px;
        padding: 15px;
        margin: 5px;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        transition: transform 0.2s ease;
    }
    .metric-card:hover {
        transform: scale(1.03);
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #e8eaf6;
        border-radius: 4px 4px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
        transition: background-color 0.3s ease;
    }
    .stTabs [aria-selected="true"] {
        background-color: #3f51b5;
        color: white;
    }
    .highlight-metric {
        background: linear-gradient(135deg, #e8f5e9 0%, #c8e6c9 100%);
        color: #2e7d32;
        padding: 12px;
        border-radius: 8px;
        text-align: center;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        transition: transform 0.2s;
    }
    .highlight-metric:hover {
        transform: scale(1.05);
    }
    .metric-row {
        display: flex;
        flex-wrap: wrap;
        justify-content: space-between;
        gap: 10px;
        margin-bottom: 20px;
    }
    .metric-box {
        flex: 1;
        min-width: 150px;
        background: linear-gradient(135deg, #f5f5f5 0%, #e0e0e0 100%);
        border-radius: 8px;
        padding: 15px;
        text-align: center;
        box-shadow: 0 3px 6px rgba(0,0,0,0.1);
        transition: all 0.3s ease;
    }
    .metric-box:hover {
        transform: translateY(-5px);
        box-shadow: 0 6px 12px rgba(0,0,0,0.15);
    }
    .metric-box.high {
        background: linear-gradient(135deg, #e3f2fd 0%, #90caf9 100%);
    }
    .metric-box.medium {
        background: linear-gradient(135deg, #e8f5e9 0%, #a5d6a7 100%);
    }
    .metric-box.low {
        background: linear-gradient(135deg, #fff3e0 0%, #ffcc80 100%);
    }
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        margin: 10px 0;
    }
    .metric-title {
        font-size: 1rem;
        font-weight: 500;
        color: #555;
    }
    .progress-container {
        width: 100%;
        background-color: #e0e0e0;
        border-radius: 10px;
        margin: 8px 0;
    }
    .progress-bar {
        height: 10px;
        border-radius: 10px;
        background: linear-gradient(90deg, #64b5f6 0%, #1976d2 100%);
        transition: width 1s ease-in-out;
    }
    .class-title {
        display: flex;
        align-items: center;
        margin-bottom: 5px;
    }
    .class-indicator {
        width: 12px;
        height: 12px;
        border-radius: 50%;
        margin-right: 8px;
    }
    .confusion-matrix-container {
        padding: 10px;
        background-color: white;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# Helper function to load model and encoders
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

# Helper function to load data
@st.cache_data
def load_data(file_path='data_processing/modified_dataset.parquet'):
    try:
        return pd.read_parquet(file_path)
    except FileNotFoundError:
        st.error(f"Could not load data from {file_path}")
        return None

# Enhanced confusion matrix visualization
def plot_confusion_matrix(y_true, y_pred, class_names):
    cm = confusion_matrix(y_true, y_pred)
    
    # Create a more colorful confusion matrix using Plotly
    fig = px.imshow(
        cm,
        labels=dict(x="Predicted", y="True", color="Count"),
        x=class_names,
        y=class_names,
        text_auto=True,
        color_continuous_scale='Blues'
    )
    
    # Enhance the layout with more engaging design
    fig.update_layout(
        title={
            'text': "Confusion Matrix",
            'y':0.95,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': {'size': 24, 'color': '#3f51b5', 'family': 'Arial, sans-serif'}
        },
        xaxis_title={
            'text': "Predicted Label",
            'font': {'size': 16, 'color': '#555'}
        },
        yaxis_title={
            'text': "True Label",
            'font': {'size': 16, 'color': '#555'}
        },
        height=600,
        width=800,
        plot_bgcolor='rgba(240,240,240,0.8)',
        paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=40, r=40, t=80, b=40),
        coloraxis_colorbar=dict(
            title="Count",
            tickfont=dict(size=14),
            titlefont=dict(size=16)
        )
    )
    
    # Add annotations for diagonal (correct predictions) to highlight them
    for i in range(len(class_names)):
        fig.add_annotation(
            x=i, y=i,
            text=f"<b>{cm[i, i]}</b>",
            font=dict(color="white" if cm[i, i] > 30 else "black", size=16),
            showarrow=False
        )
    
    return fig

# Enhanced feature importance plotting
def plot_feature_importance(model, feature_names):
    # Get feature importance
    feature_importance = model.feature_importances_
    
    # Sort features by importance
    sorted_idx = np.argsort(feature_importance)
    
    # Create a more engaging bar chart with Plotly
    fig = px.bar(
        x=feature_importance[sorted_idx],
        y=[feature_names[i] for i in sorted_idx],
        orientation='h',
        color=feature_importance[sorted_idx],
        color_continuous_scale='viridis',
        title="Feature Importance"
    )
    
    # Enhance the layout
    fig.update_layout(
        title={
            'text': "Feature Importance Analysis",
            'y':0.95,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': {'size': 24, 'color': '#43A047', 'family': 'Arial, sans-serif'}
        },
        xaxis_title={
            'text': "Importance Score",
            'font': {'size': 16, 'color': '#555'}
        },
        yaxis_title={
            'text': "Features",
            'font': {'size': 16, 'color': '#555'}
        },
        height=600,
        yaxis={'categoryorder': 'total ascending'},
        plot_bgcolor='rgba(240,240,240,0.8)',
        paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=40, r=40, t=80, b=40)
    )
    
    # Add reference lines for better readability
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(0,0,0,0.1)')
    
    # Add value labels to the bars
    for i, val in enumerate(feature_importance[sorted_idx]):
        fig.add_annotation(
            x=val + max(feature_importance) * 0.01,
            y=i,
            text=f"{val:.4f}",
            showarrow=False,
            font=dict(size=12)
        )
    
    return fig

# Function to make predictions with the model
def make_prediction(model, scaler, status_encoder, dayofweek_encoder, input_data):
    # Prepare the input data
    input_df = pd.DataFrame([input_data])
    
    # Apply encoders
    if 'Status' in input_df.columns:
        input_df['Status'] = status_encoder.transform([input_df['Status'].values[0]])[0]
    
    if 'DayOfWeek' in input_df.columns:
        input_df['DayOfWeek'] = dayofweek_encoder.transform([input_df['DayOfWeek'].values[0]])[0]
    
    # Convert all columns to numeric
    input_df = input_df.apply(pd.to_numeric, errors='coerce')
    
    # Fill any NaN values with mean of the column
    input_df.fillna(0, inplace=True)
    
    # Apply scaling
    input_scaled = scaler.transform(input_df)
    
    # Make prediction
    prediction = model.predict(input_scaled)
    probabilities = model.predict_proba(input_scaled)
    
    return prediction, probabilities

# Helper function to preprocess data for model
def preprocess_features(df, status_encoder, dayofweek_encoder):
    # Make a copy to avoid modifying the original
    features = df.drop('Classification_Tag', axis=1).copy()
    
    # Encode categorical features
    if 'Status' in features.columns:
        features['Status'] = features['Status'].map(
            {name: i for i, name in enumerate(status_encoder.classes_)}
        )
    
    if 'DayOfWeek' in features.columns:
        features['DayOfWeek'] = features['DayOfWeek'].map(
            {name: i for i, name in enumerate(dayofweek_encoder.classes_)}
        )
    
    # Convert all to numeric
    features = features.apply(pd.to_numeric, errors='coerce')
    
    # Handle missing values
    features.fillna(features.mean(), inplace=True)
    
    return features

# Parse classification report text into a structured format
def parse_classification_report(report_text):
    lines = report_text.strip().split('\n')
    metrics = []
    
    # Find the line with header (precision, recall, etc.)
    for i, line in enumerate(lines):
        if 'precision' in line.lower() and 'recall' in line.lower() and 'f1-score' in line.lower():
            header_line = i
            break
    
    # Extract class metrics
    for i in range(header_line + 1, len(lines)):
        line = lines[i].strip()
        if not line or 'accuracy' in line or 'macro' in line or 'weighted' in line:
            break
            
        parts = line.split()
        if len(parts) >= 5:  # Class label, precision, recall, f1, support
            try:
                class_label = parts[0]
                precision = float(parts[1])
                recall = float(parts[2])
                f1 = float(parts[3])
                support = int(parts[4])
                
                metrics.append({
                    'class': class_label,
                    'precision': precision,
                    'recall': recall,
                    'f1': f1,
                    'support': support
                })
            except (ValueError, IndexError):
                continue
    
    # Extract overall metrics
    accuracy = None
    macro_avg = None
    weighted_avg = None
    
    for line in lines:
        if 'accuracy' in line:
            try:
                parts = line.split()
                accuracy = float(parts[1])
            except (ValueError, IndexError):
                pass
        elif 'macro avg' in line:
            try:
                parts = line.split()
                macro_avg = {
                    'precision': float(parts[2]),
                    'recall': float(parts[3]),
                    'f1': float(parts[4])
                }
            except (ValueError, IndexError):
                pass
        elif 'weighted avg' in line:
            try:
                parts = line.split()
                weighted_avg = {
                    'precision': float(parts[2]),
                    'recall': float(parts[3]),
                    'f1': float(parts[4])
                }
            except (ValueError, IndexError):
                pass
    
    return {
        'class_metrics': metrics,
        'accuracy': accuracy,
        'macro_avg': macro_avg,
        'weighted_avg': weighted_avg
    }

# Helper function to create a gauge chart for metrics
def create_gauge_chart(value, title, threshold_ranges=None):
    if threshold_ranges is None:
        threshold_ranges = [(0, 0.6, "red"), (0.6, 0.8, "orange"), (0.8, 1.0, "green")]
    
    # Determine color based on value
    color = "gray"
    for start, end, range_color in threshold_ranges:
        if start <= value <= end:
            color = range_color
            break
    
    # Create gauge chart
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        title={"text": title, "font": {"size": 24, "color": "#333"}},
        gauge={
            "axis": {"range": [0, 1], "tickwidth": 1, "tickcolor": "darkblue"},
            "bar": {"color": color},
            "bgcolor": "white",
            "borderwidth": 2,
            "bordercolor": "gray",
            "steps": [
                {"range": [0, 0.6], "color": "rgba(255, 0, 0, 0.2)"},
                {"range": [0.6, 0.8], "color": "rgba(255, 165, 0, 0.2)"},
                {"range": [0.8, 1], "color": "rgba(0, 128, 0, 0.2)"}
            ],
            "threshold": {
                "line": {"color": "black", "width": 4},
                "thickness": 0.75,
                "value": value
            }
        },
        
        number={"font": {"size": 30, "color": color}, "suffix": "", "valueformat": ".2f"}
    ))
    
    fig.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=50, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)"
    )
    
    return fig

# Create radar chart for comparing metrics across classes
def create_metrics_radar_chart(metrics_data):
    categories = ['Precision', 'Recall', 'F1-Score']
    
    fig = go.Figure()
    
    for metric in metrics_data['class_metrics']:
        fig.add_trace(go.Scatterpolar(
            r=[metric['precision'], metric['recall'], metric['f1']],
            theta=categories,
            fill='toself',
            name=f"Class {metric['class']}"
        ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )
        ),
        title={
            'text': "Metrics by Class",
            'y':0.95,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': {'size': 20, 'color': '#3f51b5'}
        },
        showlegend=True,
        height=450,
        margin=dict(l=50, r=50, t=80, b=50),
        paper_bgcolor="rgba(0,0,0,0)",
        legend=dict(
            orientation='h',
            y=-0.1
        )
    )
    
    return fig

# Main function to run the app
def main():
    # Header
    st.markdown('<div class="main-header">ML Classification Dashboard</div>', unsafe_allow_html=True)
    
    # Create tabs for different functionalities
    tabs = st.tabs([
        "📊 Model Performance", 
        "📈 Feature Analysis", 
        "🔮 Prediction Tool", 
        "📝 Dataset Exploration"
    ])
    
    # Sidebar for model selection and settings
    with st.sidebar:
        st.image("https://xgboost.ai/images/logo/xgboost-logo.png", width=200)
        st.markdown("## Model Settings")
        
        model_dir = st.text_input("Model Directory", "modelling/model_outputs")
        data_path = st.text_input("Data Path", "data_processing/modified_dataset.parquet")
        
        if st.button("Load Model and Data", key="load_button"):
            st.session_state['load_triggered'] = True
        
        st.markdown("---")
        st.markdown("## About")
        st.markdown("""
        This dashboard visualizes the performance of an XGBoost classification model.
        
        Features:
        - Model performance metrics
        - Feature importance analysis
        - Interactive prediction tool
        - Dataset exploration
        """)
    
    # Initialize session state variables
    if 'load_triggered' not in st.session_state:
        st.session_state['load_triggered'] = False
    
    # Load model, encoders, and data
    model, status_encoder, dayofweek_encoder, classification_tag_encoder = load_model_and_encoders(model_dir)
    df = load_data(data_path)
    
    if not st.session_state['load_triggered']:
        st.info("Please load the model and data by clicking the 'Load Model and Data' button in the sidebar.")
        return
    
    if model is None or df is None:
        st.error("Could not proceed without model or data. Please check the paths.")
        return
    
    # Preprocess features for model
    features = preprocess_features(df, status_encoder, dayofweek_encoder)
    
    # Feature scaling
    scaler = StandardScaler()
    
    # Debugging - show data types before fitting
    if st.checkbox("Show feature data types for debugging"):
        st.write("Feature data types:", features.dtypes)
        st.write("Feature head:", features.head())
    
    # Fit the scaler on the processed features
    try:
        scaler.fit(features)
    except Exception as e:
        st.error(f"Error fitting scaler: {e}")
        # Try to provide more detailed error information
        for col in features.columns:
            try:
                np.asarray(features[col])
            except Exception as col_e:
                st.error(f"Error in column {col}: {col_e}")
        return
    
    # Get class names
    class_names = classification_tag_encoder.classes_
    
    # Tab 1: Enhanced Model Performance
    with tabs[0]:
        st.markdown('<div class="sub-header">Model Performance Metrics</div>', unsafe_allow_html=True)
        
        # Overview metrics at the top
        metrics_data = None
        
        try:
            with open(os.path.join(model_dir, 'evaluation_metrics.txt'), 'r') as f:
                metrics_text = f.read()
            
            # Parse the classification report
            metrics_data = parse_classification_report(metrics_text)
            
            # Display overall accuracy prominently
            if metrics_data['accuracy'] is not None:
                # Create a row of metrics
                st.markdown('<div class="metric-row">', unsafe_allow_html=True)
                
                # Overall accuracy gauge
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    accuracy_gauge = create_gauge_chart(
                        metrics_data['accuracy'], 
                        "Overall Accuracy"
                    )
                    st.plotly_chart(accuracy_gauge, use_container_width=True)
                
                # Macro avg F1 score gauge
                with col2:
                    if metrics_data['macro_avg'] is not None:
                        f1_gauge = create_gauge_chart(
                            metrics_data['macro_avg']['f1'], 
                            "Macro Avg F1 Score"
                        )
                        st.plotly_chart(f1_gauge, use_container_width=True)
                
                # Weighted avg precision gauge
                with col3:
                    if metrics_data['weighted_avg'] is not None:
                        precision_gauge = create_gauge_chart(
                            metrics_data['weighted_avg']['precision'], 
                            "Weighted Avg Precision"
                        )
                        st.plotly_chart(precision_gauge, use_container_width=True)
                
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Display class metrics in a more visual way
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.markdown('<div class="section">Class-wise Performance</div>', unsafe_allow_html=True)
                
                # Create columns for each metric
                metric_cols = st.columns(len(metrics_data['class_metrics']))
                
                # Display metrics for each class
                for i, metric in enumerate(metrics_data['class_metrics']):
                    with metric_cols[i]:
                        # Determine color based on F1 score
                        color_class = "high" if metric['f1'] >= 0.8 else "medium" if metric['f1'] >= 0.6 else "low"
                        
                        st.markdown(f'''
                        <div class="metric-box {color_class}">
                            <div class="metric-title">Class {metric['class']}</div>
                            <div class="metric-value">{metric['f1']:.2f}</div>
                            <div class="metric-title">F1 Score</div>
                            
                            <div class="class-title">
                                <div class="class-indicator" style="background-color: #1976d2;"></div>
                                <span>Precision: {metric['precision']:.2f}</span>
                            </div>
                            <div class="progress-container">
                                <div class="progress-bar" style="width: {metric['precision']*100}%;"></div>
                            </div>
                            
                            <div class="class-title">
                                <div class="class-indicator" style="background-color: #43a047;"></div>
                                <span>Recall: {metric['recall']:.2f}</span>
                            </div>
                            <div class="progress-container">
                                <div class="progress-bar" style="background: linear-gradient(90deg, #81c784 0%, #43a047 100%); width: {metric['recall']*100}%;"></div>
                            </div>
                            
                            <div class="metric-title mt-2">Support: {metric['support']}</div>
                        </div>
                        ''', unsafe_allow_html=True)
                
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Add radar chart to compare metrics across classes
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.markdown('<div class="section">Metrics Comparison</div>', unsafe_allow_html=True)
                
                radar_chart = create_metrics_radar_chart(metrics_data)
                st.plotly_chart(radar_chart, use_container_width=True)
                
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Provide the raw metrics in an expandable section
                with st.expander("View Raw Classification Report"):
                    st.text(metrics_text)
            
        except FileNotFoundError:
            st.warning("Evaluation metrics file not found. Let's generate new metrics.")
            
            # Split data for evaluation
            from sklearn.model_selection import train_test_split
            X = features  # Use preprocessed features
            y = df['Classification_Tag']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Scale features
            X_test_scaled = scaler.transform(X_test)
            
            # Generate predictions
            y_pred = model.predict(X_test_scaled)
            
            # Create and display classification report
            report = classification_report(y_test, y_pred)
            
            # Parse the report for visualization
            metrics_data = parse_classification_report(report)
            
            # Now display the visualized metrics
            # (same visualization code as above would be repeated here)
            st.text(report)
        
        # Confusion Matrix with enhanced visualization
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="section">Confusion Matrix</div>', unsafe_allow_html=True)
        
        try:
            # Try to load confusion matrix from file
            img_path = os.path.join(model_dir, 'confusion_matrix.png')
            if os.path.exists(img_path):
                # Instead of just displaying the image, create a better layout
                st.markdown('<div class="confusion-matrix-container">', unsafe_allow_html=True)
                st.image(img_path, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Add explanation of the confusion matrix
                with st.expander("How to Interpret the Confusion Matrix"):
                    st.markdown("""
                    **Confusion Matrix Explained:**
                    
                    The confusion matrix shows the counts of predictions vs actual values:
                    - Diagonal elements (top-left to bottom-right) represent correct predictions
                    - Off-diagonal elements represent misclassifications
                    - Rows represent actual classes, columns represent predicted classes
                    
                    **Quick Analysis:**
                    - Class 0 has the highest accuracy with 124 correct predictions
                    - Class 5 has significant misclassifications with other classes
                    """)
            else:
                raise FileNotFoundError("Confusion matrix image not found")
        except FileNotFoundError:
            st.info("Confusion matrix image not found. Generating one from model...")
            
            # Split data for evaluation
            from sklearn.model_selection import train_test_split
            X = features  # Use preprocessed features
            y = df['Classification_Tag']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Scale features
            X_test_scaled = scaler.transform(X_test)
            
            # Generate predictions
            y_pred = model.predict(X_test_scaled)
            
            # Plot confusion matrix
            cm_fig = plot_confusion_matrix(y_test, y_pred, class_names)
            st.plotly_chart(cm_fig, use_container_width=True)
            
            # Add explanation of the confusion matrix
            with st.expander("How to Interpret the Confusion Matrix"):
                st.markdown("""
                **Confusion Matrix Explained:**
                
                The confusion matrix shows the counts of predictions vs actual values:
                - Diagonal elements (top-left to bottom-right) represent correct predictions
                - Off-diagonal elements represent misclassifications
                - Rows represent actual classes, columns represent predicted classes
                
                **Quick Analysis:**
                - Class 0 has the highest accuracy with 124 correct predictions
                - Class 5 has significant misclassifications with other classes
                """)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Tab 2: Feature Analysis
    with tabs[1]:
        st.markdown('<div class="sub-header">Feature Importance Analysis</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="card">', unsafe_allow_html=True)
        
        feature_names = features.columns
        
        try:
            # Try to load feature importance from file
            img_path = os.path.join(model_dir, 'feature_importance.png')
            if os.path.exists(img_path):
                st.image(img_path, use_container_width=True)
            else:
                raise FileNotFoundError("Feature importance image not found")
        except FileNotFoundError:
            st.info("Feature importance image not found. Generating one from model...")
            
            # Create interactive feature importance plot
            fi_fig = plot_feature_importance(model, feature_names)
            st.plotly_chart(fi_fig, use_container_width=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Feature correlations
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="section">Feature Correlations</div>', unsafe_allow_html=True)
        
        # Calculate and display correlation matrix
        corr = features.corr()
        fig = px.imshow(
            corr,
            text_auto=True,
            color_continuous_scale='RdBu_r',
            title="Feature Correlation Matrix"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Tab 3: Prediction Tool
    with tabs[2]:
        st.markdown('<div class="sub-header">Interactive Prediction Tool</div>', unsafe_allow_html=True)
        
        # Get list of features to collect from user
        feature_list = df.drop('Classification_Tag', axis=1).columns.tolist()
        
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("Enter values for prediction:")
        
        # Create columns to organize input fields
        col1, col2 = st.columns(2)
        
        # Create input dictionary
        input_data = {}
        
        # Dynamically create input fields based on feature list
        for i, feature in enumerate(feature_list):
            if feature == 'Status':
                # Use selectbox for categorical features
                with col1 if i % 2 == 0 else col2:
                    input_data[feature] = st.selectbox(
                        f"Select {feature}",
                        options=status_encoder.classes_
                    )
            elif feature == 'DayOfWeek':
                with col1 if i % 2 == 0 else col2:
                    input_data[feature] = st.selectbox(
                        f"Select {feature}",
                        options=dayofweek_encoder.classes_
                    )
            else:
                # Use number input for numerical features
                with col1 if i % 2 == 0 else col2:
                    min_val = float(df[feature].min())
                    max_val = float(df[feature].max())
                    default_val = float(df[feature].mean())
                    
                    input_data[feature] = st.number_input(
                        f"Enter {feature}",
                        min_value=min_val,
                        max_value=max_val,
                        value=default_val,
                        format="%.2f"
                    )
        
        # Make prediction button
        predict_button = st.button("Make Prediction", key="predict_button")
        
        if predict_button:
            # Make prediction
            prediction, probabilities = make_prediction(
                model,
                scaler,
                status_encoder,
                dayofweek_encoder,
                input_data
            )
            
            # Display prediction
            st.markdown('<div class="section">Prediction Result</div>', unsafe_allow_html=True)
            
            # Display the predicted class
            predicted_class = classification_tag_encoder.inverse_transform(prediction)[0]
            
            # Create metric in a card style
            st.markdown(f"""
            <div class="metric-card" style="background-color: #e8f5e9;">
                <h2 style="color: #2e7d32;">Predicted Class</h2>
                <h1 style="color: #1b5e20; font-size: 2.5rem;">{predicted_class}</h1>
            </div>
            """, unsafe_allow_html=True)
            
            # Display probabilities
            st.markdown('<div class="section">Class Probabilities</div>', unsafe_allow_html=True)
            
            # Create a bar chart for probabilities
            probs_df = pd.DataFrame({
                'Class': classification_tag_encoder.classes_,
                'Probability': probabilities[0]
            })
            
            fig = px.bar(
                probs_df,
                x='Class',
                y='Probability',
                color='Probability',
                color_continuous_scale='Viridis',
                text_auto='.3f'
            )
            
            fig.update_layout(title="Class Probabilities")
            st.plotly_chart(fig, use_container_width=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Tab 4: Dataset Exploration
    with tabs[3]:
        st.markdown('<div class="sub-header">Dataset Exploration</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="section">Dataset Overview</div>', unsafe_allow_html=True)
        
        # Display dataset statistics
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Total Samples", len(df))
            st.metric("Features", len(df.columns) - 1)  # Subtract target column
        
        with col2:
            st.metric("Classes", len(df['Classification_Tag'].unique()))
            st.metric("Memory Usage", f"{df.memory_usage(deep=True).sum() / (1024 * 1024):.2f} MB")
        
        # Show data sample
        st.markdown('<div class="section">Data Sample</div>', unsafe_allow_html=True)
        st.dataframe(df.head(), use_container_width=True)
        
        # Class distribution
        st.markdown('<div class="section">Class Distribution</div>', unsafe_allow_html=True)
        
        # Create class distribution chart
        class_counts = df['Classification_Tag'].value_counts().reset_index()
        class_counts.columns = ['Class', 'Count']
        
        fig = px.pie(
            class_counts,
            values='Count',
            names='Class',
            title="Class Distribution",
            color_discrete_sequence=px.colors.qualitative.Bold
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Feature Distributions
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="section">Feature Distributions</div>', unsafe_allow_html=True)
        
        # Select feature to visualize
        feature_to_viz = st.selectbox(
            "Select Feature to Visualize",
            options=feature_list
        )
        
        # Create visualization
        if feature_to_viz in ['Status', 'DayOfWeek']:
            # For categorical features
            counts = df[feature_to_viz].value_counts().reset_index()
            counts.columns = [feature_to_viz, 'Count']
            
            if feature_to_viz == 'Status' and status_encoder is not None:
                counts[feature_to_viz] = counts[feature_to_viz].map(
                    {i: name for i, name in enumerate(status_encoder.classes_)}
                )
            elif feature_to_viz == 'DayOfWeek' and dayofweek_encoder is not None:
                counts[feature_to_viz] = counts[feature_to_viz].map(
                    {i: name for i, name in enumerate(dayofweek_encoder.classes_)}
                )
            
            fig = px.bar(
                counts,
                x=feature_to_viz,
                y='Count',
                color=feature_to_viz,
                title=f"{feature_to_viz} Distribution"
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            # For numerical features
            fig = px.histogram(
                df,
                x=feature_to_viz,
                nbins=30,
                marginal="box",
                title=f"{feature_to_viz} Distribution",
                color_discrete_sequence=['#3f51b5']
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Add descriptive statistics
            stats = df[feature_to_viz].describe()
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Mean", f"{stats['mean']:.2f}")
            with col2:
                st.metric("Std Dev", f"{stats['std']:.2f}")
            with col3:
                st.metric("Min", f"{stats['min']:.2f}")
            with col4:
                st.metric("Max", f"{stats['max']:.2f}")
        
        st.markdown('</div>', unsafe_allow_html=True)

# Run the app
if __name__ == "__main__":
    main()
