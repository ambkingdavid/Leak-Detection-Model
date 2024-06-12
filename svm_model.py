import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

data = pd.read_csv('data.csv')
data.head()

print(data.isnull().sum())

from imblearn.over_sampling import SMOTE

# Split the data into features and target
X = data.drop('leak_status', axis=1)
y = data['leak_status']

# Standardize the features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Handle class imbalance using SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# First split: 80% training + validation, 20% test
X_train_val, X_test, y_train_val, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled)

# Second split: 80% training, 20% validation from the training + validation set
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.25, random_state=42, stratify=y_train_val)
# 0.25 * 0.8 = 0.2 of the original data

# Verify the splits
print(f"Training set: {X_train.shape}, {y_train.shape}")
print(f"Validation set: {X_val.shape}, {y_val.shape}")
print(f"Test set: {X_test.shape}, {y_test.shape}")

from sklearn.model_selection import  StratifiedKFold, GridSearchCV, cross_val_score
from sklearn.svm import SVC

# Define the KFold cross-validator
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': [1, 0.1, 0.01, 0.001],
    'kernel': ['rbf', 'poly', 'sigmoid']
}

svm_model = SVC(probability=True)

# Initialize Grid Search with cross-validation
grid_search = GridSearchCV(estimator=svm_model, param_grid=param_grid, cv=kfold, scoring='accuracy', n_jobs=-1, verbose=2)

# Fit Grid Search
grid_search.fit(X_train, y_train)

# Best parameters and best score
best_params = grid_search.best_params_
print("Best Parameters:", best_params)
print("Best Score from Grid Search:", grid_search.best_score_)

# Train the final model using the best parameters
best_params = grid_search.best_params_
model = SVC(C=best_params['C'], gamma=best_params['gamma'], kernel=best_params['kernel'], probability=True)
model.fit(X_train, y_train)

import pickle

with open('svm_model.pkl', 'wb') as f:
    pickle.dump(model, f)


from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

with open('svm_model.pkl', 'rb') as f:
    loaded_model = pickle.load(f)

# Predict on the validation dataset
y_val_pred = loaded_model.predict(X_val)
y_val_pred_proba = loaded_model.predict_proba(X_val)  # For ROC-AUC

# Confusion Matrix for Validation set
conf_matrix_val = confusion_matrix(y_val, y_val_pred)
print("Validation Confusion Matrix:")
print(conf_matrix_val)

# Classification Report for Validation set
class_report_val = classification_report(y_val, y_val_pred, target_names=['Normal', 'Minor Leak', 'Moderate Leak', 'Severe Leak'])
print("Validation Classification Report:")
print(class_report_val)

# ROC-AUC Score for Validation set
roc_auc_val = roc_auc_score(y_val, y_val_pred_proba, multi_class='ovr')
print("Validation ROC-AUC Score:", roc_auc_val)

# Predict on the test dataset
y_test_pred = loaded_model.predict(X_test)
y_test_pred_proba = loaded_model.predict_proba(X_test)  # For ROC-AUC

# Confusion Matrix for Test set
conf_matrix_test = confusion_matrix(y_test, y_test_pred)
print("Test Confusion Matrix:")
print(conf_matrix_test)

# Classification Report for Test set
class_report_test = classification_report(y_test, y_test_pred)
print("Test Classification Report:")
print(class_report_test)

# ROC-AUC Score for Test set
roc_auc_test = roc_auc_score(y_test, y_test_pred_proba, multi_class='ovr')
print("Test ROC-AUC Score:", roc_auc_test)