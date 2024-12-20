#Feature Selection using CHI-SQUARE method
import pandas as pd
from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectKBest
from sklearn.preprocessing import LabelEncoder
# Load your dataset
df = pd.read_csv('path_to_your_dataset.csv')  # Replace with your actual file path

# Separate features and target variable
X = df.drop('target', axis=1)  # Replace 'target' with the actual name of your target variable
y = df['target']  # Replace 'target' with the actual name of your target variable

# Encode categorical variables if necessary
X_encoded = X.apply(LabelEncoder().fit_transform)
# Apply the Chi-Square test
chi2_values, p_values = chi2(X_encoded, y)

# Create a DataFrame to view the Chi-Square values and p-values
chi2_df = pd.DataFrame({'Feature': X.columns, 'Chi2 Value': chi2_values, 'P-Value': p_values})

# Sort the DataFrame by Chi-Square value in descending order
chi2_df = chi2_df.sort_values(by='Chi2 Value', ascending=False)

# Display the sorted DataFrame
print(chi2_df)
# Select the top 21 features (excluding the target variable)
top_features = chi2_df.head(21)['Feature']
print(f"Top 21 Features: {top_features}")
# Create a new DataFrame with the selected features
X_selected = X[top_features]
