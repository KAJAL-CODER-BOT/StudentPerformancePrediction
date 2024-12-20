import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
# Load your dataset
df = pd.read_csv('/content/FIINAL.csv')  # Replace with your actual file path

# Preprocess the data
df = df.dropna()  # Handle missing values by dropping rows with any missing values

# Define feature matrix X and target vector y
X = df.drop('TT', axis=1)  # Replace 'TT' with the actual name of your target variable
y = df['TT']  # Replace 'TT' with the actual name of your target variable
# Define the ABC algorithm for hyperparameter optimization
def abc_algorithm(X, y, n_iter=100, n_pop=50):
    # Initialize population
    population = np.random.rand(n_pop, 3)  # Assuming 3 hyperparameters to tune
    best_solution = None
    best_score = -np.inf

    for iteration in range(n_iter):
        for i in range(n_pop):
            # Decode the solution
            max_depth = int(population[i, 0] * 20) + 1  # Max depth between 1 and 20
            min_samples_split = int(population[i, 1] * 10) + 2  # Min samples split between 2 and 10
            min_samples_leaf = int(population[i, 2] * 10) + 1  # Min samples leaf between 1 and 10

            # Create and train the model
            model = DecisionTreeClassifier(
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                random_state=42
            )
            model.fit(X, y)
            y_pred = model.predict(X)
            score = f1_score(y, y_pred, average='weighted')

            # Update the best solution
            if score > best_score:
                best_score = score
                best_solution = (max_depth, min_samples_split, min_samples_leaf)

        # Update population (simplified for illustration)
        population = np.random.rand(n_pop, 3)

    return best_solution, best_score

# Run the ABC algorithm
best_params, best_f1 = abc_algorithm(X, y)
print(f"Best Parameters: {best_params}")
print(f"Best F1 Score: {best_f1}")
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

Expected OUTPUT:
Best Parameters: (19, 2, 1)
Best F1 Score: 0.9941698609653496
Decision Tree Metrics: {'Accuracy': 0.9824688902488781, 'Precision': 0.9828360747968404, 'Recall': 0.9824688902488781, 'F1 Score': 0.98253164116906, 'ROC AUC': 0.9827173973889678}

