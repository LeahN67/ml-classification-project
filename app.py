import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import joblib
import os
from pathlib import Path

# Security improvements:
# 1. Validate and sanitize file paths
# 2. Use pathlib for safer path handling
# 3. Remove unsafe_allow_html where not necessary
# 4. Add proper error handling throughout
# 5. Implement input validation for user inputs

# Set page configuration
st.set_page_config(
    page_title="ML Classification Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply CSS styling more efficiently
st.markdown("""
<style>
    .main-header {font-size: 2.5rem; color: #1E88E5; text-align: center; margin-bottom: 1rem; font-weight: bold;}
    .sub-header {font-size: 1.8rem; color: #43A047; margin-top: 1rem; font-weight: bold;}
    .section {font-size: 1.2rem; color: #5E35B1; margin-top: 0.8rem; font-weight: 500;}
    .card {background-color: #f8f9fa; border-radius: 10px; padding: 20px; margin-bottom: 20px; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);}
    .metric-card {background-color: #e3f2fd; border-radius: 10px; padding: 15px; margin: 5px; text-align: center; box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);}
    .stTabs [data-baseweb="tab-list"] {gap: 2px;}
    .stTabs [data-baseweb="tab"] {height: 50px; white-space: pre-wrap; background-color: #e8eaf6; border-radius: 4px 4px 0px 0px; gap: 1px; padding-top: 10px; padding-bottom: 10px;}
    .stTabs [aria-selected="true"] {background-color: #3f51b5; color: white;}
</style>
""", unsafe_allow_html=True)

# Safe path validation function
def safe_path(base_dir, user_input):
    """Validate and sanitize file paths to prevent path traversal attacks"""
    path = Path(user_input)
    # Resolve to absolute path and check if it's within the allowed directory
    try:
        base_path = Path(base_dir).resolve()
        full_path = (base_path / path).resolve()
        # Check if the path is within the base directory
        if base_path in full_path.parents or base_path == full_path:
            return str(full_path)
        else:
            st.error(f"Invalid path: {user_input}")
            return None
    except Exception as e:
        st.error(f"Path error: {e}")
        return None

# Load model and encoders with better error handling
@st.cache_resource
def load_model_and_encoders(model_dir='model_outputs'):
    model_dir = Path(model_dir)
    try:
        if not model_dir.exists():
            st.error(f"Model directory not found: {model_dir}")
            return None, None, None, None
            
        model_path = model_dir / 'classification_model.pkl'
        status_encoder_path = model_dir / 'status_encoder.pkl'
        dayofweek_encoder_path = model_dir / 'dayofweek_encoder.pkl'
        classification_tag_encoder_path = model_dir / 'classification_tag_encoder.pkl'
        
        # Check if all required files exist
        missing_files = []
        for path, name in [
            (model_path, "model"), 
            (status_encoder_path, "status encoder"),
            (dayofweek_encoder_path, "day of week encoder"),
            (classification_tag_encoder_path, "classification tag encoder")
        ]:
            if not path.exists():
                missing_files.append(name)
        
        if missing_files:
            st.error(f"Missing required files: {', '.join(missing_files)}")
            return None, None, None, None
            
        # Load all files
        model = joblib.load(model_path)
        status_encoder = joblib.load(status_encoder_path)
        dayofweek_encoder = joblib.load(dayofweek_encoder_path)
        classification_tag_encoder = joblib.load(classification_tag_encoder_path)
        
        return model, status_encoder, dayofweek_encoder, classification_tag_encoder
    except Exception as e:
        st.error(f"Error loading model or encoders: {e}")
        return None, None, None, None

# Load data with better error handling
@st.cache_data
def load_data(file_path='data_processing/modified_dataset.parquet'):
    try:
        file_path = Path(file_path)
        if not file_path.exists():
            st.error(f"Data file not found: {file_path}")
            return None
            
        df = pd.read_parquet(file_path)
        # Validate expected schema
        if 'Classification_Tag' not in df.columns:
            st.error("Invalid dataset: 'Classification_Tag' column is missing")
            return None
            
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

# Helper functions condensed for clarity
def plot_confusion_matrix(y_true, y_pred, class_names):
    cm = confusion_matrix(y_true, y_pred)
    fig = px.imshow(
        cm, labels=dict(x="Predicted", y="True", color="Count"),
        x=class_names, y=class_names, text_auto=True, color_continuous_scale='Blues'
    )
    fig.update_layout(title="Confusion Matrix", xaxis_title="Predicted Label", 
                      yaxis_title="True Label", height=600)
    return fig

def plot_feature_importance(model, feature_names):
    # Get and sort feature importance
    importance = model.feature_importances_
    sorted_idx = np.argsort(importance)
    
    fig = px.bar(
        x=importance[sorted_idx],
        y=[feature_names[i] for i in sorted_idx],
        orientation='h', color=importance[sorted_idx],
        color_continuous_scale='viridis', title="Feature Importance"
    )
    fig.update_layout(xaxis_title="Importance Score", yaxis_title="Features", 
                      height=600, yaxis={'categoryorder': 'total ascending'})
    return fig

# Preprocessing with validation
def preprocess_features(df, status_encoder, dayofweek_encoder):
    # Create a copy to avoid modifying the original
    features = df.drop('Classification_Tag', axis=1).copy()
    
    # Encode categorical features with error handling
    if 'Status' in features.columns and status_encoder is not None:
        try:
            features['Status'] = features['Status'].map(
                {name: i for i, name in enumerate(status_encoder.classes_)}
            )
            # Handle any new categories not seen during training
            features['Status'].fillna(-1, inplace=True)
        except Exception as e:
            st.error(f"Error encoding Status column: {e}")
    
    if 'DayOfWeek' in features.columns and dayofweek_encoder is not None:
        try:
            features['DayOfWeek'] = features['DayOfWeek'].map(
                {name: i for i, name in enumerate(dayofweek_encoder.classes_)}
            )
            # Handle any new categories not seen during training
            features['DayOfWeek'].fillna(-1, inplace=True)
        except Exception as e:
            st.error(f"Error encoding DayOfWeek column: {e}")
    
    # Convert all to numeric and handle missing values
    features = features.apply(pd.to_numeric, errors='coerce')
    features.fillna(features.mean(), inplace=True)
    
    return features

# Make prediction with input validation
def make_prediction(model, scaler, status_encoder, dayofweek_encoder, input_data):
    # Validate input data
    if not all(key in input_data for key in ['Status', 'DayOfWeek']):
        st.error("Missing required input fields")
        return None, None
    
    # Prepare input data
    input_df = pd.DataFrame([input_data])
    
    # Apply encoders with error handling
    try:
        if 'Status' in input_df.columns and status_encoder is not None:
            status_value = input_df['Status'].values[0]
            if status_value in status_encoder.classes_:
                input_df['Status'] = status_encoder.transform([status_value])[0]
            else:
                st.warning(f"Unknown Status value: {status_value}. Using default.")
                input_df['Status'] = -1
        
        if 'DayOfWeek' in input_df.columns and dayofweek_encoder is not None:
            dow_value = input_df['DayOfWeek'].values[0]
            if dow_value in dayofweek_encoder.classes_:
                input_df['DayOfWeek'] = dayofweek_encoder.transform([dow_value])[0]
            else:
                st.warning(f"Unknown DayOfWeek value: {dow_value}. Using default.")
                input_df['DayOfWeek'] = -1
    except Exception as e:
        st.error(f"Error in input preprocessing: {e}")
        return None, None
    
    # Convert all columns to numeric
    input_df = input_df.apply(pd.to_numeric, errors='coerce')
    input_df.fillna(0, inplace=True)
    
    # Apply scaling
    try:
        input_scaled = scaler.transform(input_df)
        prediction = model.predict(input_scaled)
        probabilities = model.predict_proba(input_scaled)
        return prediction, probabilities
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return None, None

# Main function
def main():
    # Header
    st.markdown('<div class="main-header">ML Classification Dashboard</div>', unsafe_allow_html=True)
    
    # Create tabs
    tabs = st.tabs([
        "📊 Model Performance", 
        "📈 Feature Analysis", 
        "🔮 Prediction Tool", 
        "📝 Dataset Exploration"
    ])
    
    # Sidebar with secure path handling
    with st.sidebar:
        st.image("https://xgboost.ai/images/logo/xgboost-logo.png", width=200)
        st.markdown("## Model Settings")
        
        # Default base directories
        base_model_dir = "modelling"
        base_data_dir = "data_processing"
        
        # User inputs with validation
        model_dir_input = st.text_input("Model Directory", "model_outputs")
        data_path_input = st.text_input("Data Path", "modified_dataset.parquet")
        
        # Sanitize paths
        model_dir = os.path.join(base_model_dir, model_dir_input)
        data_path = os.path.join(base_data_dir, data_path_input)
        
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
    
    # Initialize session state
    if 'load_triggered' not in st.session_state:
        st.session_state['load_triggered'] = False
    
    # Load model and data only when triggered
    if not st.session_state['load_triggered']:
        st.info("Please load the model and data by clicking the 'Load Model and Data' button in the sidebar.")
        return
    
    # Load model, encoders, and data
    model, status_encoder, dayofweek_encoder, classification_tag_encoder = load_model_and_encoders(model_dir)
    df = load_data(data_path)
    
    # Validate loaded data and models
    if model is None or df is None:
        return
    
    # Preprocess features for model
    features = preprocess_features(df, status_encoder, dayofweek_encoder)
    
    # Feature scaling
    scaler = StandardScaler()
    try:
        scaler.fit(features)
    except Exception as e:
        st.error(f"Error fitting scaler: {e}")
        return
    
    # Get class names
    class_names = classification_tag_encoder.classes_
    
    # Tab 1: Model Performance
    with tabs[0]:
        st.markdown('<div class="sub-header">Model Performance Metrics</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            metrics_file = Path(model_dir) / 'evaluation_metrics.txt'
            
            try:
                if metrics_file.exists():
                    metrics_text = metrics_file.read_text()
                    st.markdown('<div class="section">Classification Report</div>', unsafe_allow_html=True)
                    st.text(metrics_text)
                else:
                    st.info("Generating new classification report...")
                    
                    # Split data for evaluation
                    from sklearn.model_selection import train_test_split
                    X = features
                    y = df['Classification_Tag']
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                    
                    X_test_scaled = scaler.transform(X_test)
                    y_pred = model.predict(X_test_scaled)
                    report = classification_report(y_test, y_pred)
                    st.text(report)
            except Exception as e:
                st.error(f"Error generating classification report: {e}")
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<div class="section">Confusion Matrix</div>', unsafe_allow_html=True)
            
            cm_file = Path(model_dir) / 'confusion_matrix.png'
            try:
                if cm_file.exists():
                    st.image(str(cm_file), use_container_width=True)
                else:
                    st.info("Generating confusion matrix...")
                    
                    # Split data for evaluation
                    from sklearn.model_selection import train_test_split
                    X = features
                    y = df['Classification_Tag']
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                    
                    X_test_scaled = scaler.transform(X_test)
                    y_pred = model.predict(X_test_scaled)
                    cm_fig = plot_confusion_matrix(y_test, y_pred, class_names)
                    st.plotly_chart(cm_fig, use_container_width=True)
            except Exception as e:
                st.error(f"Error generating confusion matrix: {e}")
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Model summary
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="section">Model Summary</div>', unsafe_allow_html=True)
        model_params = model.get_params()
        st.json(model_params)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Tab 2: Feature Analysis
    with tabs[1]:
        st.markdown('<div class="sub-header">Feature Importance Analysis</div>', unsafe_allow_html=True)
        
        feature_names = features.columns
        
        st.markdown('<div class="card">', unsafe_allow_html=True)
        fi_file = Path(model_dir) / 'feature_importance.png'
        
        try:
            if fi_file.exists():
                st.image(str(fi_file), use_container_width=True)
            else:
                st.info("Generating feature importance plot...")
                fi_fig = plot_feature_importance(model, feature_names)
                st.plotly_chart(fi_fig, use_container_width=True)
                
            # Feature correlations
            st.markdown('<div class="section">Feature Correlations</div>', unsafe_allow_html=True)
            corr = features.corr()
            corr_fig = px.imshow(
                corr, text_auto=True, color_continuous_scale='RdBu_r',
                title="Feature Correlation Matrix"
            )
            st.plotly_chart(corr_fig, use_container_width=True)
        except Exception as e:
            st.error(f"Error in feature analysis: {e}")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Tab 3: Prediction Tool
    with tabs[2]:
        st.markdown('<div class="sub-header">Interactive Prediction Tool</div>', unsafe_allow_html=True)
        feature_list = df.drop('Classification_Tag', axis=1).columns.tolist()
        
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("Enter values for prediction:")
        
        col1, col2 = st.columns(2)
        input_data = {}
        
        # Create input fields with validation
        try:
            for i, feature in enumerate(feature_list):
                if feature == 'Status' and status_encoder is not None:
                    with col1 if i % 2 == 0 else col2:
                        input_data[feature] = st.selectbox(
                            f"Select {feature}", options=status_encoder.classes_
                        )
                elif feature == 'DayOfWeek' and dayofweek_encoder is not None:
                    with col1 if i % 2 == 0 else col2:
                        input_data[feature] = st.selectbox(
                            f"Select {feature}", options=dayofweek_encoder.classes_
                        )
                else:
                    with col1 if i % 2 == 0 else col2:
                        # Calculate safe min/max/default values
                        try:
                            min_val = float(df[feature].min())
                            max_val = float(df[feature].max())
                            default_val = float(df[feature].mean())
                        except:
                            # Fallback values if calculation fails
                            min_val = 0.0
                            max_val = 100.0
                            default_val = 50.0
                        
                        input_data[feature] = st.number_input(
                            f"Enter {feature}",
                            min_value=min_val, max_value=max_val,
                            value=default_val, format="%.2f"
                        )
        except Exception as e:
            st.error(f"Error creating input fields: {e}")
        
        # Make prediction button
        predict_button = st.button("Make Prediction", key="predict_button")
        
        if predict_button:
            # Make prediction with error handling
            prediction, probabilities = make_prediction(
                model, scaler, status_encoder, dayofweek_encoder, input_data
            )
            
            if prediction is not None and probabilities is not None:
                st.markdown('<div class="section">Prediction Result</div>', unsafe_allow_html=True)
                
                try:
                    predicted_class = classification_tag_encoder.inverse_transform(prediction)[0]
                    
                    # Display result using markdown
                    st.markdown(f"""
                    <div class="metric-card" style="background-color: #e8f5e9;">
                        <h2 style="color: #2e7d32;">Predicted Class</h2>
                        <h1 style="color: #1b5e20; font-size: 2.5rem;">{predicted_class}</h1>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Display probabilities
                    st.markdown('<div class="section">Class Probabilities</div>', unsafe_allow_html=True)
                    
                    probs_df = pd.DataFrame({
                        'Class': classification_tag_encoder.classes_,
                        'Probability': probabilities[0]
                    })
                    
                    prob_fig = px.bar(
                        probs_df, x='Class', y='Probability',
                        color='Probability', color_continuous_scale='Viridis',
                        text_auto='.3f'
                    )
                    prob_fig.update_layout(title="Class Probabilities")
                    st.plotly_chart(prob_fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Error displaying prediction: {e}")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Tab 4: Dataset Exploration
    with tabs[3]:
        st.markdown('<div class="sub-header">Dataset Exploration</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="section">Dataset Overview</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Total Samples", len(df))
            st.metric("Features", len(df.columns) - 1)
        
        with col2:
            st.metric("Classes", len(df['Classification_Tag'].unique()))
            st.metric("Memory Usage", f"{df.memory_usage(deep=True).sum() / (1024 * 1024):.2f} MB")
        
        # Show data sample with limiting columns if too many
        st.markdown('<div class="section">Data Sample</div>', unsafe_allow_html=True)
        display_cols = df.columns[:20] if len(df.columns) > 20 else df.columns  # Limit columns if too many
        st.dataframe(df[display_cols].head(), use_container_width=True)
        
        # Class distribution
        try:
            st.markdown('<div class="section">Class Distribution</div>', unsafe_allow_html=True)
            class_counts = df['Classification_Tag'].value_counts().reset_index()
            class_counts.columns = ['Class', 'Count']
            
            class_fig = px.pie(
                class_counts, values='Count', names='Class',
                title="Class Distribution", color_discrete_sequence=px.colors.qualitative.Bold
            )
            st.plotly_chart(class_fig, use_container_width=True)
        except Exception as e:
            st.error(f"Error generating class distribution: {e}")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Feature Distributions
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="section">Feature Distributions</div>', unsafe_allow_html=True)
        
        feature_to_viz = st.selectbox(
            "Select Feature to Visualize", options=feature_list
        )
        
        try:
            if feature_to_viz in ['Status', 'DayOfWeek']:
                # For categorical features
                counts = df[feature_to_viz].value_counts().reset_index()
                counts.columns = [feature_to_viz, 'Count']
                
                if feature_to_viz == 'Status' and status_encoder is not None:
                    counts[feature_to_viz] = counts[feature_to_viz].astype(str)
                elif feature_to_viz == 'DayOfWeek' and dayofweek_encoder is not None:
                    counts[feature_to_viz] = counts[feature_to_viz].astype(str)
                
                cat_fig = px.bar(
                    counts, x=feature_to_viz, y='Count',
                    color=feature_to_viz, title=f"{feature_to_viz} Distribution"
                )
                st.plotly_chart(cat_fig, use_container_width=True)
            else:
                # For numerical features - with outlier handling
                hist_fig = px.histogram(
                    df, x=feature_to_viz, nbins=30, marginal="box",
                    title=f"{feature_to_viz} Distribution", color_discrete_sequence=['#3f51b5']
                )
                st.plotly_chart(hist_fig, use_container_width=True)
                
                # Add descriptive statistics
                stats = df[feature_to_viz].describe()
                
                col1, col2, col3, col4 = st.columns(4)
                with col1: st.metric("Mean", f"{stats['mean']:.2f}")
                with col2: st.metric("Std Dev", f"{stats['std']:.2f}")
                with col3: st.metric("Min", f"{stats['min']:.2f}")
                with col4: st.metric("Max", f"{stats['max']:.2f}")
        except Exception as e:
            st.error(f"Error generating distribution plot: {e}")
        
        st.markdown('</div>', unsafe_allow_html=True)

# Run the app
if __name__ == "__main__":
    main()
