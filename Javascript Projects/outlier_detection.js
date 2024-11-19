// Example dataset
const data = [10, 12, 12, 13, 12, 11, 12, 14, 100, 12, 13, 12, 11];

// Function to calculate the mean manually
function calculateMean(data) {
    // Sum all data points
    const sumOfData = data.reduce((acc, val) => acc + val, 0);

    // Number of data points
    const N = data.length;

    // Mean is the sum divided by the number of data points
    const mean = sumOfData / N;

    return mean;
}

// Function to calculate the standard deviation manually
function calculateStdDev(data, mean) {
    // Number of data points
    const N = data.length;

    // Calculate the sum of squared differences from the mean
    const squaredDiffs = data.map(x => Math.pow(x - mean, 2));

    // Sum of squared differences
    const sumOfSquaredDiffs = squaredDiffs.reduce((acc, val) => acc + val, 0);

    // Calculate variance (for sample standard deviation use N-1)
    const variance = sumOfSquaredDiffs / (N - 1);

    // Standard deviation is the square root of variance
    const stdDev = Math.sqrt(variance);

    return stdDev;
}

// Compute the mean manually
const meanManual = calculateMean(data);
console.log(`Manually Calculated Mean: ${meanManual}`);

// Compute the standard deviation manually using the mean calculated above
const stdDevManual = calculateStdDev(data, meanManual);
console.log(`Manually Calculated Standard Deviation: ${stdDevManual}`);

// Compute the z-scores using the manually calculated mean and standard deviation
const zScores = data.map(x => (x - meanManual) / stdDevManual);
console.log(`Z-scores: ${zScores}`);

// Set a threshold for z-scores to consider as outliers
const threshold = 3;

// Identify outliers
const outliers = data.filter((x, i) => Math.abs(zScores[i]) > threshold);
console.log(`Outliers: ${outliers}`);
