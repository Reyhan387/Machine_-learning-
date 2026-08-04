import pandas as pd
import numpy as np

# Read Dataset
df = pd.read_csv("Car data.csv")

# Display Top 5 Records
print(df.head())

# Display Last 5 Records
print(df.tail())

# Dataset Information
print(df.info())

# Statistical Summary
print(df.describe())

# Display Column Names
print(df.columns)

# Extract Brand and Module from Car Name
print(df.head())

# Unique Brand Names
print(df["Brand"].unique())

# Number of Unique Brands
print(df["Brand"].nunique())

# Unique Module Names
print(df["Module"].unique())

# Statistical Summary (Numeric Columns)
print(df.describe().T)

# Statistical Summary (All Columns)
print(df.describe(include="all").T)

# Fill missing values with 0
print(df.fillna(0, inplace=True))