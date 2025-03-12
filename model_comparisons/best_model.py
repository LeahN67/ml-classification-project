import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.utils import resample, class_weight
from sklearn.metrics import classification_report
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier

# Define output folder
output_folder = r"C:\Users\HP\Documents\ml-classification-project\data_processing\best_model\best_model_outputs"
os.makedirs(output_folder, exist_ok=True)

# Load dataset
df = pd.read_parquet(r"C:\Users\HP\Documents\ml-classification-project\data_processing\modified_dataset.parquet")

# Encode categorical variables
label_encoders = {}
for col in ['Status', 'DayOfWeek', 'Classification_Tag']:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Define features and target
X = df.drop('Classification_Tag', axis=1)
y = df['Classification_Tag']

# Handle class imbalance with upsampling
df_resampled = pd.concat([X, y], axis=1)
majority_class = df_resampled[df_resampled['Classification_Tag'] == df_resampled['Classification_Tag'].mode()[0]]
minority_classes = [df_resampled[df_resampled['Classification_Tag'] == i] 
                    for i in df_resampled['Classification_Tag'].unique() 
                    if i != df_resampled['Classification_Tag'].mode()[0]]
upsampled_minority_classes = [resample(minority_class, replace=True, n_samples=len(majority_class), random_state=42) 
                              for minority_class in minority_classes]
df_upsampled = pd.concat([majority_class] + upsampled_minority_classes)

X_resampled = df_upsampled.drop('Classification_Tag', axis=1)
y_resampled = df_upsampled['Classification_Tag']

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled)

# Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Define models and hyperparameter grids
models = {
    'XGBoost': XGBClassifier(random_state=42),
    'LightGBM': LGBMClassifier(random_state=42, min_gain_to_split=0.01),
    'RandomForest': RandomForestClassifier(random_state=42)
}

param_grids = {
    'XGBoost': {'n_estimators': [50, 100, 200], 'max_depth': [3, 6, 10], 'learning_rate': [0.01, 0.1, 0.2]},
    'LightGBM': {'n_estimators': [50, 100, 200], 'num_leaves': [31, 50, 100], 'learning_rate': [0.01, 0.1, 0.2], 'min_gain_to_split': [0.01, 0.1]},
    'RandomForest': {'n_estimators': [50, 100, 200], 'max_depth': [None, 10, 20]}
}

best_model = None
best_score = 0
best_report = None

# Perform hyperparameter tuning and model selection
for name, model in models.items():
    random_search = RandomizedSearchCV(model, param_distributions=param_grids[name], n_iter=20, scoring='f1_weighted', cv=3, random_state=42, n_jobs=1)
    random_search.fit(X_train, y_train)
    y_pred = random_search.best_estimator_.predict(X_test)
    report = classification_report(y_test, y_pred)
    score = random_search.best_score_
    
    print(f"{name} Classification Report:\n", report)
    if score > best_score:
        best_model = random_search.best_estimator_
        best_score = score
        best_report = report

# Save best model and evaluation report

joblib.dump(best_model, os.path.join(output_folder, "best_model.pkl"))
with open(os.path.join(output_folder, "evaluation_report.txt"), "w") as f:
    f.write(best_report)

print("Best Model saved to:", output_folder)
print("Best Classification Report:\n", best_report)
