# Sample data
X = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
y = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]

# Initialize coefficients
b0 = 0
b1 = 0
learning_rate = 0.1
num_iterations = 1000


# Sigmoid function
def sigmoid(z):
    exp_z = 1 + (2.718281828459045 ** -z)  # Using an approximation of e
    return 1 / exp_z


# Logistic regression model prediction
def predict(x):
    return sigmoid(b0 + b1 * x)


# Gradient descent algorithm using binary cross-entropy loss
for _ in range(num_iterations):
    # Initialize gradients
    gradient_b0 = 0
    gradient_b1 = 0

    # Compute gradients
    for i in range(len(X)):
        y_pred = predict(X[i])
        error = y_pred - y[i]
        gradient_b0 += error
        gradient_b1 += error * X[i]

    # Update coefficients
    b0 -= learning_rate * gradient_b0 / len(X)
    b1 -= learning_rate * gradient_b1 / len(X)

# Make predictions
y_pred = [1 if predict(x) >= 0.5 else 0 for x in X]

# Calculate accuracy
correct_predictions = sum([1 if y[i] == y_pred[i] else 0 for i in range(len(y))])
accuracy = correct_predictions / len(y)

# Print results
print("Coefficients: b0 =", b0, ", b1 =", b1)
print("Accuracy:", accuracy)

# Calculate binary cross-entropy loss
import math

def binary_cross_entropy(y_true, y_pred):
    epsilon = 1e-15  # to avoid log(0)
    y_pred = max(min(y_pred, 1 - epsilon), epsilon)
    return - (y_true * math.log(y_pred) + (1 - y_true) * math.log(1 - y_pred))


total_loss = sum([binary_cross_entropy(y[i], predict(X[i])) for i in range(len(y))]) / len(y)
print("Binary Cross-Entropy Loss:", total_loss)
