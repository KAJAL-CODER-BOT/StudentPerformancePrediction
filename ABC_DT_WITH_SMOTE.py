import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from imblearn.over_sampling import SMOTE
from abc import ABC, abstractmethod
# Load your dataset
df = pd.read_csv('/content/FIINAL.csv')  # Replace with your actual file path

# Preprocess the data
df = df.dropna()  # Handle missing values by dropping rows with any missing values

# Define feature matrix X and target vector y
X = df.drop('TT', axis=1)  # Replace 'TT' with the actual name of your target variable
y = df['TT']  # Replace 'TT' with the actual name of your target variable
class ArtificialBeeColony(ABC):
    @abstractmethod
    def optimize(self, X, y):
        pass
class ABC_DecisionTree(ArtificialBeeColony):
    def __init__(self, population_size=10, max_iter=100):
        self.population_size = population_size
        self.max_iter = max_iter

    def optimize(self, X, y):
        # Initialize population
        population = np.random.rand(self.population_size, 3)  # Example: 3 parameters to tune
        best_solution = None
        best_score = -np.inf

        for iteration in range(self.max_iter):
            for individual in population:
                # Decode individual to parameters
                max_depth = int(individual[0] * 20) + 1  # Example: max_depth between 1 and 20
                min_samples_split = int(individual[1] * 10) + 2  # Example: min_samples_split between 2 and 10
                min_samples_leaf = int(individual[2] * 10) + 1  # Example: min_samples_leaf between 1 and 10

                # Create and train the model
                model = DecisionTreeClassifier(
                    max_depth=max_depth,
                    min_samples_split=min_samples_split,
                    min_samples_leaf=min_samples_leaf,
                    random_state=42
                )

                # Apply SMOTE to handle class imbalance
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
                    best_solution = (max_depth, min_samples_split, min_samples_leaf)

        return best_solution, best_score
# Initialize the ABC algorithm
abc = ABC_DecisionTree(population_size=10, max_iter=100)

# Optimize hyperparameters
best_params, best_f1_score = abc.optimize(X.values, y.values)

print(f"Best Hyperparameters: {best_params}")
print(f"Best F1 Score: {best_f1_score}")
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

# Define the model with the best parameters
model = DecisionTreeClassifier(
    max_depth=best_params[0],
    min_samples_split=best_params[1],
    min_samples_leaf=best_params[2],
    random_state=42
)

# Evaluate the model
metrics = evaluate_model(model, X, y, cv)
print(f"Decision Tree Metrics: {metrics}")

OUTPUT:
Best Hyperparameters: (15, 10, 2)
Best F1 Score: 0.9545116258627286
Decision Tree Metrics: {'Accuracy': 0.9400754793961648, 'Precision': 0.9451473452268012, 'Recall': 0.9400754793961648, 'F1 Score': 0.9408783157058775, 'ROC AUC': 0.9808964379417434}
