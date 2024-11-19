// Sample data
// X: Feature, y: Target
const X = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
const y = [3, 4, 2, 5, 6, 7, 8, 9, 10, 11];

// Function to calculate mean
function mean(values) {
    return values.reduce((sum, value) => sum + value, 0) / values.length;
}

// Function to calculate variance
function variance(values, meanValue) {
    return values.reduce((sum, value) => sum + (value - meanValue) ** 2, 0);
}

// Function to calculate covariance
function covariance(x, meanX, y, meanY) {
    return x.reduce((sum, xi, i) => sum + (xi - meanX) * (y[i] - meanY), 0);
}

// Calculate the mean of X and y
const meanX = mean(X);
const meanY = mean(y);

// Calculate variance and covariance
const varX = variance(X, meanX);
const covarXY = covariance(X, meanX, y, meanY);

// Calculate coefficients
const b1 = covarXY / varX;
const b0 = meanY - b1 * meanX;

// Make predictions
function predict(x) {
    return b0 + b1 * x;
}

// Predictions for the sample data
const yPred = X.map(x => predict(x));

// Calculate Mean Squared Error
const mse = y.reduce((sum, yi, i) => sum + (yi - yPred[i]) ** 2, 0) / y.length;

// Calculate R^2 score
const ssTotal = y.reduce((sum, yi) => sum + (yi - meanY) ** 2, 0);
const ssResidual = y.reduce((sum, yi, i) => sum + (yi - yPred[i]) ** 2, 0);
const r2Score = 1 - (ssResidual / ssTotal);

console.log(`Coefficients: b0 = ${b0}, b1 = ${b1}`);
console.log(`Mean squared error: ${mse}`);
console.log(`Coefficient of determination (R^2): ${r2Score}`);

// Plot outputs using Plotly.js
const trace1 = {
    x: X,
    y: y,
    mode: 'markers',
    type: 'scatter',
    name: 'Data Points'
};

const trace2 = {
    x: X,
    y: yPred,
    mode: 'lines',
    type: 'scatter',
    name: 'Fitted Line'
};

const layout = {
    xaxis: { title: 'X' },
    yaxis: { title: 'y' },
    title: 'Linear Regression Example'
};

Plotly.newPlot('plot', [trace1, trace2], layout);
