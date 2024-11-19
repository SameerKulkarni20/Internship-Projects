# Sample data
X = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
y = [3, 4, 2, 5, 6, 7, 8, 9, 10, 11]

# Function to calculate mean
def mean(values):
    return sum(values) / len(values)

# Function to calculate variance
def variance(values, mean_value):
    return sum([(x - mean_value) ** 2 for x in values]) / (len(values) - 1)  # Dividing by N - 1

# Function to calculate sample covariance
def covariance(x, mean_x, y, mean_y):
    covar = 0.0
    for i in range(len(x)):
        covar += (x[i] - mean_x) * (y[i] - mean_y)
    return covar / (len(x) - 1)  # Dividing by N - 1

# Calculate the mean of X and y
mean_x = mean(X)
mean_y = mean(y)

# Calculate variance and covariance
var_x = variance(X, mean_x)
covar_xy = covariance(X, mean_x, y, mean_y)

# Calculate coefficients
b1 = covar_xy / var_x
b0 = mean_y - b1 * mean_x

# Make predictions
def predict(x):
    return b0 + b1 * x

# Predictions for the sample data
y_pred = [predict(x) for x in X]

# Calculate Mean Squared Error
mse = sum([(y[i] - y_pred[i]) ** 2 for i in range(len(y))]) / len(y)

# Calculate R^2 score
ss_total = sum([(yi - mean_y) ** 2 for yi in y])
ss_residual = sum([(y[i] - y_pred[i]) ** 2 for i in range(len(y))])
r2_score = 1 - (ss_residual / ss_total)

print("Coefficients: b0 =", b0, ", b1 =", b1)
print("Mean squared error:", mse)
print("Coefficient of determination (R^2):", r2_score)

# Plot outputs
import matplotlib.pyplot as plt

plt.scatter(X, y, color='black')
plt.plot(X, y_pred, color='blue', linewidth=3)

plt.xlabel('X')
plt.ylabel('y')
plt.title('Linear Regression Example')
plt.show()
