# Implementation of z-score and outlier detection

# Example dataset
data = [10, 12, 12, 13, 12, 11, 12, 14, 100, 12, 13, 12, 11]

# To Calculate the mean manually
def calculate_mean(data):
    # To make the Sum all data points
    sum_of_data = sum(data)

    # Number of particular data points
    N = len(data)

    # Mean is the sum divided by the number of data points
    mean = sum_of_data / N

    return mean


# To Calculate the standard deviation manually
def calculate_std_dev(data, mean):
    # Number of data points
    N = len(data)

    # To Calculate the sum of squared differences from the mean
    squared_diffs = [(x - mean) ** 2 for x in data]

    # Find the Sum of squared differences
    sum_of_squared_diffs = sum(squared_diffs)

    # Calculate variance (for sample standard deviation use N-1)
    variance = sum_of_squared_diffs / (N - 1)

    # Standard deviation is the square root of variance
    std_dev = variance**0.5

    return std_dev


# Compute the mean manually
mean_manual = calculate_mean(data)
print(f"Manually Calculated Mean: {mean_manual}")

# Compute the standard deviation manually using the mean calculated above
std_dev_manual = calculate_std_dev(data, mean_manual)
print(f"Manually Calculated Standard Deviation: {std_dev_manual}")

# Compute the z-scores using the manually calculated mean and standard deviation
z_scores = [(x - mean_manual) / std_dev_manual for x in data]
print(f"Z-scores: {z_scores}")

# Set a threshold for z-scores to consider as outliers
threshold = 3

# Identify outliers
outliers = [data[i] for i in range(len(data)) if abs(z_scores[i]) > threshold]

# Print the results
print(f"Outliers: {outliers}")
