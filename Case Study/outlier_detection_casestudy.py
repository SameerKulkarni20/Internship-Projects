import pandas as pd
import numpy as np

# Load the dataset
data = pd.read_excel("/Users/sameerkulkarni/Downloads/ecommerce_transactions_complex.xlsx")

# Function to calculate the mean manually
def calculate_mean(column):
    sum_of_data = sum(column)
    N = len(column)
    mean = sum_of_data / N
    return mean

# Function to calculate the standard deviation manually
def calculate_std_dev(column, mean):
    N = len(column)
    squared_diffs = [(x - mean) ** 2 for x in column]
    sum_of_squared_diffs = sum(squared_diffs)
    variance = sum_of_squared_diffs / (N - 1)
    std_dev = variance ** 0.5
    return std_dev

# Filter only numeric columns
numeric_data = data.select_dtypes(include=[np.number])

# Calculate mean and standard deviation for each numeric column manually
means = numeric_data.apply(calculate_mean, axis=0)
std_devs = numeric_data.apply(lambda x: calculate_std_dev(x, means[x.name]), axis=0)

# Calculate z-scores for each numeric column manually
z_scores = numeric_data.apply(lambda x: (x - means[x.name]) / std_devs[x.name], axis=0)

# Set a threshold for z-scores to consider as outliers
threshold = 3

# Identify outliers
outliers = data[(np.abs(z_scores) > threshold).any(axis=1)]

# Print the results with better alignment
print("Means:")
for column, value in means.items():
    print(f"{column:<20} {value:>10.6f}")

print("-" * 24)

print("Standard Deviations:")
for column, value in std_devs.items():
    print(f"{column:<20} {value:>10.6f}")

print("-" * 24)

print("Z-scores:")
print(z_scores)

print("-" * 24)

print("Outliers:")
print(outliers)


