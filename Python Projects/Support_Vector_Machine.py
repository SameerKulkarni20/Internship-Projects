# Sample data (1D for simplicity)
X = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
y = [-1, -1, -1, -1, -1, 1, 1, 1, 1, 1]  # Labels: -1 or 1

# Initialize weights and bias
w = 0
b = 0
learning_rate = 0.001
lambda_param = 0.01  # Regularization parameter
num_iterations = 1000

# Function to compute hinge loss and its gradient
def hinge_loss_gradient(w, b, X, y):
    dw = 0
    db = 0
    for i in range(len(X)):
        if y[i] * (w * X[i] + b) < 1:
            dw += -y[i] * X[i]
            db += -y[i]
    dw = dw / len(X) + lambda_param * w
    db = db / len(X)
    return dw, db

# Gradient descent to optimize weights and bias
for _ in range(num_iterations):
    dw, db = hinge_loss_gradient(w, b, X, y)
    w -= learning_rate * dw
    b -= learning_rate * db

# Make predictions
def predict(x):
    return 1 if w * x + b >= 0 else -1

# Testing the model
y_pred = [predict(x) for x in X]
correct_predictions = sum(1 for i in range(len(y)) if y[i] == y_pred[i])
accuracy = correct_predictions / len(y)

# Print results
print("Weight:", w)
print("Bias:", b)
print("Accuracy:", accuracy)
