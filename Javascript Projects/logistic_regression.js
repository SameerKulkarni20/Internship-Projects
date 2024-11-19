// Sample data
const X = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
const y = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1];

// Initialize coefficients
let b0 = 0;
let b1 = 0;
const learningRate = 0.1;
const numIterations = 1000;

// Sigmoid function
function sigmoid(z) {
    return 1 / (1 + Math.exp(-z));
}

// Logistic regression model prediction
function predict(x) {
    return sigmoid(b0 + b1 * x);
}

// Gradient descent algorithm using binary cross-entropy loss
for (let iteration = 0; iteration < numIterations; iteration++) {
    // Initialize gradients
    let gradientB0 = 0;
    let gradientB1 = 0;

    // Compute gradients
    for (let i = 0; i < X.length; i++) {
        const yPred = predict(X[i]);
        const error = yPred - y[i];
        gradientB0 += error;
        gradientB1 += error * X[i];
    }

    // Update coefficients
    b0 -= learningRate * gradientB0 / X.length;
    b1 -= learningRate * gradientB1 / X.length;
}

// Make predictions
const yPred = X.map(x => (predict(x) >= 0.5 ? 1 : 0));

// Calculate accuracy
const correctPredictions = y.reduce((count, yi, i) => count + (yi === yPred[i] ? 1 : 0), 0);
const accuracy = correctPredictions / y.length;

// Print results
console.log(`Coefficients: b0 = ${b0}, b1 = ${b1}`);
console.log(`Accuracy: ${accuracy}`);

// Calculate binary cross-entropy loss
function binaryCrossEntropy(yTrue, yPred) {
    const epsilon = 1e-15;  // to avoid log(0)
    yPred = Math.max(Math.min(yPred, 1 - epsilon), epsilon);
    return - (yTrue * Math.log(yPred) + (1 - yTrue) * Math.log(1 - yPred));
}

const totalLoss = y.reduce((sum, yi, i) => sum + binaryCrossEntropy(yi, predict(X[i])), 0) / y.length;
console.log(`Binary Cross-Entropy Loss: ${totalLoss}`);
