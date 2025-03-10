# ML Classification Dashboard

This repository contains a machine learning classification project that includes **data exploration**, **data processing**, **model building**, and an **interactive dashboard** for predictions. The project focuses on classifying financial transactions using a combination of rule-based and machine learning approaches.

![Dashboard Preview](assets/dashboard_preview.png)

---

## Project Structure

```
ml-classification-project/
│
├── eda/                   # Exploratory Data Analysis scripts
│   └── transaction_analysis_dashboard.py    # Script for exploratory data analysis
│
├── data_processing/       # Data processing scripts
│   └── data_processing.py  # Script for processing the transactional dataset
│   └── modified_dataset.parquet  # Processed dataset
│
├── rule_based_classification/  # Rule-based classification scripts
│   └── transaction_classifier.py  # Script for rule-based classification approach
│
├── modelling/             # Model building scripts
│   └── xgboost_script.py     # Script for training the XGBoost model
│   └── model_outputs/     # Directory for saving model artifacts
│       ├── classification_model.pkl
│       ├── status_encoder.pkl
│       ├── dayofweek_encoder.pkl
│       ├── classification_tag_encoder.pkl
│       ├── evaluation_metrics.txt
│       ├── confusion_matrix.png
│       └── feature_importance.png
│
├── app.py                 # Streamlit dashboard application
├── requirements.txt       # Project dependencies
└── README.md              # Project documentation
```

---

## Getting Started

### Prerequisites

- **Python 3.8+**
- **Git**
- **Streamlit** (for running the dashboard)

### Setup Instructions

1. **Clone the repository**

   ```bash
   git clone https://github.com/yourusername/ml-classification-project.git
   cd ml-classification-project
   ```

2. **Create a Python virtual environment**

   ```bash
   # Using venv
   python -m venv <environment_name>

   # Activate the environment:
   # On Windows
   <environment_name>\Scripts\activate
   # On macOS/Linux
   source <environment_name>/bin/activate
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

---

## Project Workflow

### 1. **Exploratory Data Analysis (EDA)**

Run the EDA script to analyze the transactional dataset:

```bash
python eda/transaction_analysis_dashboard.py
```

This script analyzes the dataset to understand:
- **Data distribution**
- **Feature correlations**
- **Missing values**
- **Class distribution**
- **Statistical summaries**

---

### 2. **Data Processing**

Process the raw transactional data:

```bash
python data_processing/data_processing.py
```

This step performs:
- **Feature engineering**
- **Handling missing values**
- **Encoding categorical variables**
- **Removing outliers**
- **Normalization/standardization**
- Saving the processed dataset as `modified_dataset.parquet`

---

### 3. **Classification Approaches**

#### **Option 1: Rule-Based Classification (Optional)**

You can first run a rule-based classification algorithm before training a machine learning model:

```bash
python rule_based_classification/transaction_classifier.py
```

This approach:
- Implements a set of predefined business rules
- Classifies data based on explicit conditions
- Can serve as a baseline or interpretable model
- Provides insights that can inform feature engineering

#### **Option 2: XGBoost Model**

Train the XGBoost classification model:

```bash
python modelling/xgboost_script.py
```

This script:
- Loads the processed data
- Splits it into training and test sets
- Trains an XGBoost classifier
- Evaluates the model performance
- Saves the model and encoders for future use

---

### 4. **Running the Dashboard**

Launch the Streamlit dashboard to visualize model performance and make predictions:

```bash
streamlit run app.py
```

The dashboard provides:
- **Model performance metrics**
- **Feature importance analysis**
- **Interactive prediction tool**
- **Dataset exploration capabilities**

---

## Dashboard Features

### **Model Performance**
- Classification report with precision, recall, and F1 score
- Interactive confusion matrix
- Model parameters summary

### **Feature Analysis**
- Feature importance visualization
- Feature correlation heatmap
- Detailed feature insights

### **Prediction Tool**
- Interactive interface for inputting feature values
- Real-time prediction with class probabilities
- Visual representation of prediction results

### **Dataset Exploration**
- Dataset overview and statistics
- Class distribution visualization
- Feature distribution analysis

---

## Customization

You can customize the model directory and data path in the dashboard's sidebar:

1. Click on the sidebar icon (if collapsed).
2. Enter your custom model directory path.
3. Enter your custom data path.
4. Click **"Load Model and Data"** to apply changes.

**To Access Dashboard:**https://ml-classification.streamlit.app/

---

## Dependencies

Key libraries used in this project:

| Library         | Purpose                                   |
|-----------------|-------------------------------------------|
| **streamlit**   | For building the interactive dashboard    |
| **pandas**      | For data manipulation                    |
| **numpy**       | For numerical operations                 |
| **scikit-learn**| For model evaluation and preprocessing   |
| **xgboost**     | For the classification model             |
| **matplotlib**  | For static visualizations                |
| **seaborn**     | For enhanced visualizations              |
| **plotly**      | For interactive visualizations           |

---

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository.
2. Create a new branch for your feature or bugfix.
3. Commit your changes with clear and descriptive messages.
4. Submit a Pull Request (PR) with a detailed description of your changes.

---

## Acknowledgments

- **Streamlit** for providing an excellent framework for building interactive dashboards.
- **XGBoost** for its powerful and efficient implementation of gradient boosting.
- **Scikit-learn** for its comprehensive suite of machine learning tools.

---

This README provides a comprehensive guide to the project, including setup instructions, workflow details, and customization options. Let me know if you need further assistance!