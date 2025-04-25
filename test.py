import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

# File Path
file_path = 'Dataset/1 spring-framework/2015-6.csv'

# Check if the dataset file exists
if not os.path.exists(file_path):
    print(f"Error: File '{file_path}' not found. Please check the file path.")
    exit()

# Load dataset
dataset = pd.read_csv(file_path)

# Function to plot column distributions
def plotPerColumnDistribution(df, nGraphShown=10, nGraphPerRow=5):
    nunique = df.nunique()
    df = df[[col for col in df if 1 < nunique[col] < 50]]  # Filter columns with 1-50 unique values
    
    nRow, nCol = df.shape
    columnNames = list(df)
    nGraphRow = (nCol + nGraphPerRow - 1) // nGraphPerRow  # Use integer division

    plt.figure(figsize=(6 * nGraphPerRow, 8 * nGraphRow), dpi=80)
    
    for i in range(min(nCol, nGraphShown)):
        plt.subplot(nGraphRow, nGraphPerRow, i + 1)
        columnDf = df.iloc[:, i]
        if not np.issubdtype(type(columnDf.iloc[0]), np.number):
            columnDf.value_counts().plot.bar()
        else:
            columnDf.hist()
        plt.ylabel('Counts')
        plt.xticks(rotation=90)
        plt.title(f'{columnNames[i]} (Column {i})')
    
    plt.tight_layout()
    plt.show()

# Handling missing values
dataset.fillna(0, inplace=True)

# Check if required columns exist before applying Label Encoding
cols = ['QualifiedName', 'Name', 'Complexity', 'Coupling', 'Size', 'Lack of Cohesion']
missing_cols = [col for col in cols if col not in dataset.columns]

if missing_cols:
    print(f"Warning: Missing columns {missing_cols} in the dataset.")
    exit()

# Apply Label Encoding
le = LabelEncoder()
for col in cols:
    dataset[col] = le.fit_transform(dataset[col].astype(str))

# Extract 'Complexity' as target variable Y before dropping
if 'Complexity' in dataset.columns:
    Y = dataset['Complexity'].values
    dataset.drop(['Complexity'], axis=1, inplace=True)
else:
    print("Error: 'Complexity' column not found in dataset.")
    exit()

# Convert dataset to numpy array
X = dataset.values

# Debugging prints
print("Dataset processed successfully.")
print(f"Shape of X: {X.shape}, Shape of Y: {Y.shape}")

# Optional: Plot feature distributions
# plotPerColumnDistribution(dataset, 10, 5)
