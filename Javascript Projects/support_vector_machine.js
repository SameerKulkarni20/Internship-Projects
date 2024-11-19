// Sample data (1D for simplicity)
const X = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
const y = [-1, -1, -1, -1, -1, 1, 1, 1, 1, 1];  // Labels: -1 or 1

// Initialize weights and bias
let w = 0;
let b = 0;
const learningRate = 0.001;
const lambdaParam = 0.01;  // Regularization parameter
const numIterations = 1000;

// Function to compute hinge loss and its gradient
function hingeLossGradient(w, b, X, y) {
    let dw = 0;
    let db = 0;
    for (let i = 0; i < X.length; i++) {
        if (y[i] * (w * X[i] + b) < 1) {
            dw += -y[i] * X[i];
            db += -y[i];
        }
    }
    dw = dw / X.length + lambdaParam * w;
    db = db / X.length;
    return { dw, db };
}

// Gradient descent to optimize weights and bias
for (let iteration = 0; iteration < numIterations; iteration++) {
    const { dw, db } = hingeLossGradient(w, b, X, y);
    w -= learningRate * dw;
    b -= learningRate * db;
}

// Make predictions
function predict(x) {
    return (w * x + b >= 0) ? 1 : -1;
}

// Testing the model
const yPred = X.map(x => predict(x));
const correctPredictions = y.reduce((count, yi, i) => count + (yi === yPred[i] ? 1 : 0), 0);
const accuracy = correctPredictions / y.length;

// Print results
console.log(`Weight: ${w}`);
console.log(`Bias: ${b}`);
console.log(`Accuracy: ${accuracy}`);
