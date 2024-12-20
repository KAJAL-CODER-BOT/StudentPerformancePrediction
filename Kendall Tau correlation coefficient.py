#Kendall Tau correlation coefficient
#FEATURE COEFFICIENT CORRELATION USING KENDALL TAU
import pandas as pd

# Load your dataset
dataframe1 = pd.read_csv("/content/PANAGAR_MULTI_CSV.csv")
# Adjust display settings to show all features in output
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)
# Compute Kendall Tau correlation coefficient
kendall_corr = dataframe1.corr(method='kendall')

# Print the correlation matrix
print("Kendall Tau Correlation Matrix:")
print(kendall_corr)

