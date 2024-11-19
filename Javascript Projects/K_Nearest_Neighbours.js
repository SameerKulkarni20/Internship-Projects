//Javascript Code for KNN

function distance(p1, p2) {
    let dist = 0;
    for (let i = 0; i < p1.length; i++) {
        dist += Math.pow(p1[i] - p2[i], 2);
    }
    return Math.sqrt(dist);
}

// K-Nearest Neighbors algorithm
function knn(X_train, y_train, X_test, k) {
    let predictions = [];

    X_test.forEach(test_point => {
        // Calculate distances from the test point to all training points
        let distances = [];

        for (let i = 0; i < X_train.length; i++) {
            let dist = distance(test_point, X_train[i]);
            distances.push([dist, y_train[i]]);
        }

        // Sort distances and get the k nearest labels
        distances.sort((a, b) => a[0] - b[0]);
        let k_nearest = distances.slice(0, k).map(pair => pair[1]);

        // Majority vote
        let prediction = k_nearest.sort((a, b) =>
            k_nearest.filter(v => v === a).length
            - k_nearest.filter(v => v === b).length
        ).pop();

        predictions.push(prediction);
    });

    return predictions;
}

// Sample data
let X_train = [[1, 2], [2, 3], [3, 3], [6, 6], [7, 8], [8, 8]];
let y_train = [0, 0, 0, 1, 1, 1];
let X_test = [[4, 4], [5, 5], [1, 1], [7, 7]];

// Number of neighbors
let k = 3;

// Predictions
let predictions = knn(X_train, y_train, X_test, k);
console.log("Predicted classes:", predictions);
