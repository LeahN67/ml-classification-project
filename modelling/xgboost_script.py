import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.utils import resample, class_weight
import matplotlib.pyplot as plt
import joblib
import os
from xgboost import XGBClassifier
import numpy as np

# Define the output folder
output_folder = r'C:\Users\HP\Documents\credrails\modelling\model_outputs'
os.makedirs(output_folder, exist_ok=True)

# Load the dataset
file_path = r'data_processing/modified_dataset.parquet'  # Update the path if needed
df = pd.read_parquet(file_path)

# Create and save label encoders
status_encoder = LabelEncoder()
dayofweek_encoder = LabelEncoder()
classification_tag_encoder = LabelEncoder()

# Fit and transform categorical columns
df['Status'] = status_encoder.fit_transform(df['Status'])
df['DayOfWeek'] = dayofweek_encoder.fit_transform(df['DayOfWeek'])
df['Classification_Tag'] = classification_tag_encoder.fit_transform(df['Classification_Tag'])

# Save label encoders
joblib.dump(status_encoder, os.path.join(output_folder, 'status_encoder.pkl'))
joblib.dump(dayofweek_encoder, os.path.join(output_folder, 'dayofweek_encoder.pkl'))
joblib.dump(classification_tag_encoder, os.path.join(output_folder, 'classification_tag_encoder.pkl'))

# Retrieve class names
class_names = classification_tag_encoder.classes_

# Split the data into features and target
X = df.drop('Classification_Tag', axis=1)
y = df['Classification_Tag']

# Handle class imbalance using upsampling
df_resampled = pd.concat([X, y], axis=1)

# Separate majority and minority classes
majority_class = df_resampled[df_resampled['Classification_Tag'] == df_resampled['Classification_Tag'].mode()[0]]
minority_classes = [df_resampled[df_resampled['Classification_Tag'] == i] 
                    for i in df_resampled['Classification_Tag'].unique() 
                    if i != df_resampled['Classification_Tag'].mode()[0]]

# Upsample minority classes
upsampled_minority_classes = [resample(minority_class, replace=True, 
                                       n_samples=len(majority_class), random_state=42) 
                              for minority_class in minority_classes]

# Combine majority class with upsampled minority classes
df_upsampled = pd.concat([majority_class] + upsampled_minority_classes)

# Split the data into features and target
X_resampled = df_upsampled.drop('Classification_Tag', axis=1)
y_resampled = df_upsampled['Classification_Tag']

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, 
                                                    test_size=0.2, random_state=42, stratify=y_resampled)

# Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Calculate class weights
class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weights_dict = dict(enumerate(class_weights))

# Model training with XGBoost and RandomizedSearchCV
model = XGBClassifier(random_state=42, eval_metric='mlogloss')
param_dist = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 6, 10],
    'learning_rate': [0.01, 0.1, 0.2],
    'subsample': [0.5, 0.7, 1.0],
    'colsample_bytree': [0.5, 0.7, 1.0]
}

random_search = RandomizedSearchCV(model, param_distributions=param_dist, n_iter=50, 
                                   scoring='f1_weighted', cv=3, random_state=42, n_jobs=-1)
random_search.fit(X_train, y_train, sample_weight=np.vectorize(class_weights_dict.get)(y_train))
model = random_search.best_estimator_

# Predictions and evaluation
y_pred = model.predict(X_test)

# Generate classification report with exact class labels
classification_rep = classification_report(y_test, y_pred, target_names=class_names)
conf_matrix = confusion_matrix(y_test, y_pred)

# Visualization of the confusion matrix
plt.figure(figsize=(8, 6))
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=class_names)
disp.plot(cmap='Blues', ax=plt.gca())
plt.title('Confusion Matrix')
plt.savefig(os.path.join(output_folder, 'confusion_matrix.png'))

# Feature importance
feature_importances = model.feature_importances_
plt.figure(figsize=(10, 6))
plt.barh(X.columns, feature_importances, color='skyblue')
plt.xlabel('Feature Importance')
plt.title('Feature Importance for XGBoost Classifier')
plt.tight_layout()
plt.savefig(os.path.join(output_folder, 'feature_importance.png'))

# Save the model and metrics
joblib.dump(model, os.path.join(output_folder, 'classification_model.pkl'))
with open(os.path.join(output_folder, 'evaluation_metrics.txt'), 'w') as f:
    f.write(classification_rep)

print('Model, metrics, and label encoders saved successfully to the folder:', output_folder)
