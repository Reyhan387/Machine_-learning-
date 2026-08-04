import pandas as pd
import numpy as np

# Read Dataset
df = pd.read_csv("Car data.csv")

# Print Complete Dataset
print(df)

# Check Duplicate Values
print("Duplicate Rows :", df.duplicated().sum())

# Check Missing Values
print("Missing Values :")
print(df.isnull().sum())

# Display Data Types
print("Data Types :")
print(df.dtypes)