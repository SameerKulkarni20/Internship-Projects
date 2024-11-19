import matplotlib
import pandas as pd
import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt

# Read the CSV file
print("Reading the CSV file...")
df = pd.read_csv('/Users/sameerkulkarni/Downloads/canada_per_capita_income.csv')
print("DataFrame:")
print(df)

# Plot the data
print("Plotting the data...")
plt.xlabel('year')
plt.ylabel('per capita income (US$)')
plt.scatter(df['year'], df['per capita income (US$)'], color='red', marker='+')
plt.show()

# Drop the 'per capita income (US$)' column to get the feature set
print("Dropping the 'per capita income (US$)' column...")
new_df = df.drop('per capita income (US$)', axis=1)
print("New DataFrame:")
print(new_df)

# Extract the target variable
print("Extracting the target variable...")
pci = df['per capita income (US$)']
print("Per capita income:")
print(pci)

# Create linear regression object
print("Creating and training the linear regression model...")
reg = linear_model.LinearRegression()
reg.fit(new_df, pci)

# Predict the per capita income for the year 2020
print("Predicting the per capita income for the year 2020...")
predicted_income = reg.predict([[2020]])
print(f"Predicted per capita income for the year 2020: {predicted_income[0]}")
