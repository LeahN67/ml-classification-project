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

# Custom CSS to make the dashboard more colorful and professional
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
        font-weight: bold;
    }
    .sub-header {
        font-size: 1.8rem;
        color: #43A047;
        margin-top: 1rem;
        font-weight: bold;
    }
    .section {
        font-size: 1.2rem;
        color: #5E35B1;
        margin-top: 0.8rem;
        font-weight: 500;
    }
    .card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .metric-card {
        background-color: #e3f2fd;
        border-radius: 10px;
        padding: 15px;
        margin: 5px;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
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
    }
    .stTabs [aria-selected="true"] {
        background-color: #3f51b5;
        color: white;
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

# Helper function to create confusion matrix visualization
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
    
    fig.update_layout(
        title="Confusion Matrix",
        xaxis_title="Predicted Label",
        yaxis_title="True Label",
        height=600,
        width=800
    )
    
    return fig

# Helper function to plot feature importance
def plot_feature_importance(model, feature_names):
    # Get feature importance
    feature_importance = model.feature_importances_
    
    # Sort features by importance
    sorted_idx = np.argsort(feature_importance)
    
    # Create a bar chart with Plotly
    fig = px.bar(
        x=feature_importance[sorted_idx],
        y=[feature_names[i] for i in sorted_idx],
        orientation='h',
        color=feature_importance[sorted_idx],
        color_continuous_scale='viridis',
        title="Feature Importance"
    )
    
    fig.update_layout(
        xaxis_title="Importance Score",
        yaxis_title="Features",
        height=600,
        yaxis={'categoryorder': 'total ascending'}
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

# Main function to run the app
def main():
    
    # Header with custom styling
    st.markdown("""
    <style>
        .main-header {
            font-size: 2.5rem;
            font-weight: 700;
            color: #1E88E5;
            text-align: center;
            margin-bottom: 2rem;
            padding: 1rem;
            border-bottom: 2px solid #90CAF9;
        }
        .sub-header {
            font-size: 1.8rem;
            font-weight: 600;
            color: #0D47A1;
            margin: 1rem 0;
        }
        .section {
            font-size: 1.3rem;
            font-weight: 500;
            color: #1565C0;
            margin: 0.8rem 0;
        }
        .card {
            background-color: #FFFFFF;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            padding: 1.5rem;
            margin-bottom: 1.5rem;
        }
        .metric-value {
            font-size: 2rem;
            font-weight: 700;
            color: #1E88E5;
        }
        .metric-label {
            font-size: 1rem;
            color: #546E7A;
        }
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
        }
        .stTabs [data-baseweb="tab"] {
            background-color: #E3F2FD;
            border-radius: 4px 4px 0 0;
            padding: 10px 16px;
            font-weight: 500;
        }
        .stTabs [aria-selected="true"] {
            background-color: #90CAF9 !important;
            color: #0D47A1 !important;
        }
        .sidebar-content {
            padding: 1rem;
            background-color: #F5F7FA;
            border-radius: 8px;
        }
        .footer {
            text-align: center;
            color: #546E7A;
            margin-top: 3rem;
            font-size: 0.9rem;
        }
        /* Button styling */
        .stButton>button {
            background-color: #1E88E5;
            color: white;
            border-radius: 5px;
            padding: 0.5rem 1rem;
            font-weight: 500;
            border: none;
            width: 100%;
        }
        .stButton>button:hover {
            background-color: #0D47A1;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
        }
        /* Error message styling */
        .stAlert {
            border-radius: 8px;
        }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="main-header">ML Classification Dashboard</div>', unsafe_allow_html=True)
    
    # Create tabs with consistent icons and improved styling
    tabs = st.tabs([
        "📊 Model Performance", 
        "📈 Feature Analysis", 
        "🔮 Prediction Tool", 
        "📝 Dataset Exploration"
    ])
    
    # Enhanced sidebar with better organization and styling
    with st.sidebar:
        # Logo with proper padding and sizing
        col1, col2, col3 = st.columns([1, 3, 1])
        with col2:
            st.image("https://xgboost.ai/images/logo/xgboost-logo.png", width=150)
        
        st.markdown('<div class="sidebar-content">', unsafe_allow_html=True)
        st.markdown("## Model Settings")
        
        # Input fields with better descriptions and defaults
        model_dir = st.text_input(
            "Model Directory", 
            "modelling/model_outputs",
            help="Directory containing the trained model files and evaluation metrics"
        )
        data_path = st.text_input(
            "Data Path", 
            "data_processing/modified_dataset.parquet",
            help="Path to the processed dataset for analysis"
        )
        
        # More prominent button with loading state
        load_button = st.button(
            "📂 Load Model and Data", 
            key="load_button",
            help="Click to load the selected model and dataset"
        )
        
        if load_button:
            with st.spinner('Loading model and data...'):
                st.session_state['load_triggered'] = True
                st.success('✅ Model and data loaded successfully!')
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        # About section with better styling
        st.markdown('<div class="sidebar-content" style="margin-top: 20px;">', unsafe_allow_html=True)
        st.markdown("## About")
        st.markdown("""
        This dashboard visualizes the performance of an XGBoost classification model with interactive features.
        
        ### Features:
        - 📊 Comprehensive model performance metrics
        - 📈 In-depth feature importance analysis
        - 🔮 Interactive prediction tool
        - 📝 Detailed dataset exploration
        
        *Built with Streamlit and XGBoost*
        """)
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Version info
        st.markdown('<div class="footer">Version 1.2.0</div>', unsafe_allow_html=True)
    
    # Initialize session state variables
    if 'load_triggered' not in st.session_state:
        st.session_state['load_triggered'] = False
    if 'features' not in st.session_state:
        st.session_state['features'] = None
    if 'model' not in st.session_state:
        st.session_state['model'] = None
    if 'scaler' not in st.session_state:
        st.session_state['scaler'] = None
    if 'df' not in st.session_state:
        st.session_state['df'] = None
    if 'class_names' not in st.session_state:
        st.session_state['class_names'] = None
    
    # Load model, encoders, and data
    if st.session_state['load_triggered']:
        with st.spinner('Preparing environment...'):
            model, status_encoder, dayofweek_encoder, classification_tag_encoder = load_model_and_encoders(model_dir)
            df = load_data(data_path)
            
            # Store in session state
            st.session_state['model'] = model
            st.session_state['df'] = df
        
            if model is None or df is None:
                st.error("⚠️ Could not proceed without model or data. Please check the paths and try again.")
            else:
                # Preprocess features for model
                with st.spinner('Processing features...'):
                    features = preprocess_features(df, status_encoder, dayofweek_encoder)
                    # Store features in session state
                    st.session_state['features'] = features
                
                    # Feature scaling
                    scaler = StandardScaler()
                    
                    # Debug toggle with improved styling
                    with st.expander("Debug Information", expanded=False):
                        debug_toggle = st.checkbox("Show feature data types", value=False)
                        if debug_toggle:
                            st.write("Feature data types:", features.dtypes)
                            st.dataframe(features.head(), use_container_width=True)
                    
                    # Fit the scaler on the processed features
                    try:
                        scaler.fit(features)
                        # Store scaler in session state
                        st.session_state['scaler'] = scaler
                    except Exception as e:
                        st.error(f"⚠️ Error fitting scaler: {e}")
                        # Try to provide more detailed error information
                        error_found = False
                        for col in features.columns:
                            try:
                                np.asarray(features[col])
                            except Exception as col_e:
                                st.error(f"Error in column {col}: {col_e}")
                                error_found = True
                        
                        if not error_found:
                            st.error("No specific column errors found. Please check your data preprocessing steps.")
                        return
                    
                    # Get class names
                    class_names = classification_tag_encoder.classes_
                    # Store class names in session state
                    st.session_state['class_names'] = class_names
                    
                    # Get features from session state to ensure it's available globally
                    features = st.session_state['features']
                    model = st.session_state['model']
                    scaler = st.session_state['scaler']
                    df = st.session_state['df']
                    class_names = st.session_state['class_names']
                    
                    # Define feature_names here to ensure it's available
                    feature_names = features.columns if features is not None else []
                    
                    # Tab 1: Model Performance with enhanced visuals
                    with tabs[0]:
                        st.markdown('<div class="sub-header">Model Performance Metrics</div>', unsafe_allow_html=True)
                        
                        # Key metrics summary before detailed view
                        metric_cols = st.columns(4)
                        try:
                            # Try to extract key metrics like accuracy, precision, recall, f1
                            metrics_file = os.path.join(model_dir, 'evaluation_metrics.txt')
                            if os.path.exists(metrics_file):
                                with open(metrics_file, 'r') as f:
                                    metrics_text = f.read()
                                    
                                # Extract accuracy from the text (assuming it's in the format)
                                import re
                                acc_match = re.search(r'accuracy\s*:\s*(0\.\d+)', metrics_text)
                                precision_match = re.search(r'weighted avg\s*\d+\s*(0\.\d+)', metrics_text)
                                recall_match = re.search(r'weighted avg\s*\d+\s*0\.\d+\s*(0\.\d+)', metrics_text)
                                f1_match = re.search(r'weighted avg\s*\d+\s*0\.\d+\s*0\.\d+\s*(0\.\d+)', metrics_text)
                                
                                accuracy = float(acc_match.group(1)) if acc_match else None
                                precision = float(precision_match.group(1)) if precision_match else None
                                recall = float(recall_match.group(1)) if recall_match else None
                                f1 = float(f1_match.group(1)) if f1_match else None
                            else:
                                # Generate metrics
                                from sklearn.model_selection import train_test_split
                                from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
                                
                                X = features
                                y = df['Classification_Tag']
                                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                                
                                X_test_scaled = scaler.transform(X_test)
                                y_pred = model.predict(X_test_scaled)
                                
                                accuracy = accuracy_score(y_test, y_pred)
                                precision = precision_score(y_test, y_pred, average='weighted')
                                recall = recall_score(y_test, y_pred, average='weighted')
                                f1 = f1_score(y_test, y_pred, average='weighted')
                            
                            # Display key metrics in cards
                            with metric_cols[0]:
                                st.markdown('<div class="card">', unsafe_allow_html=True)
                                st.markdown('<div class="metric-label">Accuracy</div>', unsafe_allow_html=True)
                                st.markdown(f'<div class="metric-value">{accuracy:.2%}</div>', unsafe_allow_html=True)
                                st.markdown('</div>', unsafe_allow_html=True)
                                
                            with metric_cols[1]:
                                st.markdown('<div class="card">', unsafe_allow_html=True)
                                st.markdown('<div class="metric-label">Precision</div>', unsafe_allow_html=True)
                                st.markdown(f'<div class="metric-value">{precision:.2%}</div>', unsafe_allow_html=True)
                                st.markdown('</div>', unsafe_allow_html=True)
                                
                            with metric_cols[2]:
                                st.markdown('<div class="card">', unsafe_allow_html=True)
                                st.markdown('<div class="metric-label">Recall</div>', unsafe_allow_html=True)
                                st.markdown(f'<div class="metric-value">{recall:.2%}</div>', unsafe_allow_html=True)
                                st.markdown('</div>', unsafe_allow_html=True)
                                
                            with metric_cols[3]:
                                st.markdown('<div class="card">', unsafe_allow_html=True)
                                st.markdown('<div class="metric-label">F1 Score</div>', unsafe_allow_html=True)
                                st.markdown(f'<div class="metric-value">{f1:.2%}</div>', unsafe_allow_html=True)
                                st.markdown('</div>', unsafe_allow_html=True)
                                
                        except Exception as e:
                            st.warning(f"Could not extract or calculate metrics: {e}")
                        
                        # Detailed metrics and confusion matrix
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown('<div class="card">', unsafe_allow_html=True)
                            try:
                                with open(os.path.join(model_dir, 'evaluation_metrics.txt'), 'r') as f:
                                    metrics_text = f.read()
                                
                                st.markdown('<div class="section">Classification Report</div>', unsafe_allow_html=True)
                                
                                # Format the classification report for better readability
                                st.code(metrics_text, language="text")
                            except FileNotFoundError:
                                st.warning("Evaluation metrics file not found. Generating new metrics...")
                                
                                # Split data for evaluation
                                from sklearn.model_selection import train_test_split
                                from sklearn.metrics import classification_report
                                
                                X = features  # Use preprocessed features
                                y = df['Classification_Tag']
                                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                                
                                # Scale features
                                X_test_scaled = scaler.transform(X_test)
                                
                                # Generate predictions
                                y_pred = model.predict(X_test_scaled)
                                
                                # Create and display classification report
                                report = classification_report(y_test, y_pred)
                                st.code(report, language="text")
                            st.markdown('</div>', unsafe_allow_html=True)
                        
                        with col2:
                            st.markdown('<div class="card">', unsafe_allow_html=True)
                            st.markdown('<div class="section">Confusion Matrix</div>', unsafe_allow_html=True)
                            
                            try:
                                # Try to load confusion matrix from file
                                img_path = os.path.join(model_dir, 'confusion_matrix.png')
                                if os.path.exists(img_path):
                                    st.image(img_path, use_container_width=True)
                                else:
                                    raise FileNotFoundError("Confusion matrix image not found")
                            except FileNotFoundError:
                                st.info("Confusion matrix image not found. Generating a new visualization...")
                                
                                # Split data for evaluation
                                from sklearn.model_selection import train_test_split
                                X = features  # Use preprocessed features
                                y = df['Classification_Tag']
                                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                                
                                # Scale features
                                X_test_scaled = scaler.transform(X_test)
                                
                                # Generate predictions
                                y_pred = model.predict(X_test_scaled)
                                
                                # Plot confusion matrix with improved styling
                                cm_fig = plot_confusion_matrix(y_test, y_pred, class_names)
                                st.plotly_chart(cm_fig, use_container_width=True)
                            
                            st.markdown('</div>', unsafe_allow_html=True)
                        
                        # Model summary with collapsible sections
                        st.markdown('<div class="card">', unsafe_allow_html=True)
                        st.markdown('<div class="section">Model Configuration</div>', unsafe_allow_html=True)
                        
                        model_params = model.get_params()
                        # Group parameters for better organization
                        param_groups = {
                            "Core Parameters": ["n_estimators", "learning_rate", "max_depth", "booster"],
                            "Tree Parameters": ["min_child_weight", "gamma", "subsample", "colsample_bytree"],
                            "Learning Parameters": ["objective", "eval_metric", "early_stopping_rounds"],
                            "Other Parameters": []  # Will contain all other parameters
                        }
                        
                        # Sort parameters into groups
                        for param in model_params:
                            found = False
                            for group, group_params in param_groups.items():
                                if param in group_params:
                                    found = True
                                    break
                            if not found:
                                param_groups["Other Parameters"].append(param)
                        
                        # Create tabs for parameter groups
                        param_tabs = st.tabs(list(param_groups.keys()))
                        
                        for i, (group, params) in enumerate(param_groups.items()):
                            with param_tabs[i]:
                                if params:  # Check if the group has any parameters
                                    for param in params:
                                        if param in model_params:
                                            st.markdown(f"**{param}:** `{model_params[param]}`")
                                else:
                                    st.info(f"No parameters in the {group} group.")
                        
                        st.markdown('</div>', unsafe_allow_html=True)
    else:
        # Welcome screen when no data is loaded
        st.markdown('<div class="card" style="text-align: center; padding: 3rem;">', unsafe_allow_html=True)
        st.image("https://xgboost.ai/images/logo/xgboost-logo.png", width=200)
        st.markdown('<div style="font-size: 1.5rem; margin: 1rem 0;">Welcome to the ML Classification Dashboard</div>', unsafe_allow_html=True)
        st.markdown("""
        Please load your model and data to get started:
        1. Verify the model directory path in the sidebar
        2. Verify the data path in the sidebar
        3. Click the 'Load Model and Data' button
        """)
        st.markdown('<div style="margin: 2rem 0;">', unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Set features to None if not loaded yet - ensures no errors when accessing features.columns
        features = None
        st.session_state['features'] = None
    
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
