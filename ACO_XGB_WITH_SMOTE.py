import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from imblearn.over_sampling import SMOTE
# Load your dataset
df = pd.read_csv('/content/FIINAL.csv')  # Replace with your actual file path

# Preprocess the data
df = df.dropna()  # Handle missing values by dropping rows with any missing values

# Define feature matrix X and target vector y
X = df.drop('TT', axis=1)  # Replace 'TT' with the actual name of your target variable
y = df['TT']  # Replace 'TT' with the actual name of your target variable
class AntColonyOptimization:
    def __init__(self, population_size=10, max_iter=100):
        self.population_size = population_size
        self.max_iter = max_iter

    def optimize(self, X, y):
        # Initialize population
        population = np.random.rand(self.population_size, 2)  # Example: 2 parameters to tune
        best_solution = None
        best_score = -np.inf

        for iteration in range(self.max_iter):
            for individual in population:
                # Decode individual to parameters
                n_estimators = int(individual[0] * 100) + 1  # Example: n_estimators between 1 and 100
                learning_rate = individual[1] * 0.1  # Example: learning_rate between 0 and 0.1

                # Create and train the model
                model = XGBClassifier(
                    n_estimators=n_estimators,
                    learning_rate=learning_rate,
                    random_state=42
                )

                # Apply SMOTE to balance the dataset
                smote = SMOTE(random_state=42)
                X_res, y_res = smote.fit_resample(X, y)

                # Evaluate the model using cross-validation
                cv = StratifiedKFold(n_splits=8, shuffle=True, random_state=42)
                scores = []
                for train_idx, test_idx in cv.split(X_res, y_res):
                    X_train, X_test = X_res[train_idx], X_res[test_idx]
                    y_train, y_test = y_res[train_idx], y_res[test_idx]
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    scores.append(f1_score(y_test, y_pred))

                mean_score = np.mean(scores)
                if mean_score > best_score:
                    best_score = mean_score
                    best_solution = (n_estimators, learning_rate)

        return best_solution, best_score
# Initialize the ACO algorithm
aco = AntColonyOptimization(population_size=10, max_iter=100)

# Optimize hyperparameters
best_params, best_f1_score = aco.optimize(X, y)

print(f"Best Hyperparameters: {best_params}")
print(f"Best F1 Score: {best_f1_score}")
def evaluate_model(model, X, y, cv):
    accuracy_scores = []
    precision_scores = []
    recall_scores = []
    f1_scores = []
    roc_auc_scores = []

    for train_idx, test_idx in cv.split(X, y):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        model.fit(X_train, y_train)
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
# Define the XGBoost model with the best parameters
model = XGBClassifier(
    n_estimators=best_params[0],
    learning_rate=best_params[1],
    random_state=42
)

# Initialize StratifiedKFold for 8-fold cross-validation
cv = StratifiedKFold(n_splits=8, shuffle=True, random_state=42)

# Evaluate the model
metrics = evaluate_model(model, X, y, cv)
print(f"XGBoost Metrics: {metrics}")
