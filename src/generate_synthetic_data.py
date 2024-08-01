import os
import pandas as pd
import numpy as np

# Set a random seed for reproducibility
np.random.seed(42)

# Generate synthetic data
n_samples = 1000
feature1 = np.random.randn(n_samples)  # Example feature 1
feature2 = np.random.randn(n_samples)  # Example feature 2
target = np.random.randint(0, 2, size=n_samples)  # Binary target variable

# Create a DataFrame
data = pd.DataFrame({
    'feature1': feature1,
    'feature2': feature2,
    'target': target
})

# Ensure the 'data' directory exists
os.makedirs('data', exist_ok=True)

# Save the synthetic data to a CSV file
data.to_csv('data/transactions.csv', index=False)

