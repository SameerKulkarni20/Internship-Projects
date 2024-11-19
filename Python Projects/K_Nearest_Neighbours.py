# Function to calculate Euclidean distance
def distance(p1, p2):
    dist = 0
    for i in range(len(p1)):
        dist += (p1[i] - p2[i]) ** 2
    return dist ** 0.5

# K-Nearest Neighbors algorithm
def knn(X_train, y_train, X_test, k):
    predictions = []
    for test_point in X_test:
        # Calculate distances from the test point to all training points
        distances = []
        for i in range(len(X_train)):
            dist = distance(test_point, X_train[i])
            distances.append((dist, y_train[i]))
        
        # Sort distances and get the k nearest labels
        distances.sort(key=lambda x: x[0])
        k_nearest = [label for _, label in distances[:k]]
        
        # Majority vote
        prediction = max(set(k_nearest), key=k_nearest.count)
        predictions.append(prediction)
    
    return predictions

# Sample data
X_train = [[1, 2], [2, 3], [3, 3], [6, 6], [7, 8], [8, 8]]
y_train = [0, 0, 0, 1, 1, 1]
X_test = [[4, 4], [5, 5], [1, 1], [7, 7]]

# Number of neighbors
k = 3

# Predictions
predictions = knn(X_train, y_train, X_test, k)
print("Predicted classes:", predictions)
