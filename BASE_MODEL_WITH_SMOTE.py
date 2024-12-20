# Install necessary libraries
!pip install pandas numpy scikit-learn xgboost imbalanced-learn

# Import libraries
import pandas as pd
import numpy as np
import warnings
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from imblearn.over_sampling import SMOTE
warnings.filterwarnings('ignore')

# Load your dataset
df = pd.read_csv('/content/FIINAL.csv')  # Replace with your actual file path

# Preprocess the data
df = df.dropna()  # Handle missing values by dropping rows with any missing values

# Define feature matrix X and target vector y
X = df.drop('TT', axis=1)  # Replace 'target_column' with the actual name of your target variable
y = df['TT']  # Replace 'target_column' with the actual name of your target variable

# Initialize models
dt_model = DecisionTreeClassifier(random_state=42)
knn_model = KNeighborsClassifier()
xgb_model = XGBClassifier(eval_metric='mlogloss', random_state=42)

# Initialize StratifiedKFold for 8-fold cross-validation
cv = StratifiedKFold(n_splits=8, shuffle=True, random_state=42)

# Function to evaluate models
def evaluate_model(model, X, y, cv):
    accuracy_scores = []
    precision_scores = []
    recall_scores = []
    f1_scores = []
    roc_auc_scores = []

    for train_idx, test_idx in cv.split(X, y):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        # Apply SMOTE to the training data
        smote = SMOTE(random_state=42)
        X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

        model.fit(X_train_res, y_train_res)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

        accuracy_scores.append(np.mean(y_pred == y_test))
        precision_scores.append(precision_score(y_test, y_pred, average='weighted'))
        recall_scores.append(recall_score(y_test, y_pred, average='weighted'))
        f1_scores.append(f1_score(y_test, y_pred, average='weighted'))
        roc_auc_scores.append(roc_auc_score(y_test, y_proba) if y_proba is not None else np.nan)

    return {
        'Accuracy': np.mean(accuracy_scores),
        'Precision': np.mean(precision_scores),
        'Recall': np.mean(recall_scores),
        'F1 Score': np.mean(f1_scores),
        'ROC AUC': np.mean(roc_auc_scores)
    }

# Evaluate Decision Tree
dt_metrics = evaluate_model(dt_model, X, y, cv)
print(f"Decision Tree Metrics: {dt_metrics}")

# Evaluate K-Nearest Neighbors
knn_metrics = evaluate_model(knn_model, X, y, cv)
print(f"K-Nearest Neighbors Metrics: {knn_metrics}")

# Evaluate XGBoost
xgb_metrics = evaluate_model(xgb_model, X, y, cv)
print(f"XGBoost Metrics: {xgb_metrics}")


