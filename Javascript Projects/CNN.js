// Simple Convolution, ReLU, and Pooling in a compact form

// Example 5x5 Image and 3x3 Kernel
const image = [
    [1, 2, 0, 3, 1],
    [4, 0, 1, 1, 0],
    [1, 3, 2, 2, 4],
    [0, 1, 1, 3, 1],
    [2, 4, 1, 0, 0],
];

const kernel = [
    [1, 0, -1],
    [1, 0, -1],
    [1, 0, -1],
];

// Helper function to calculate the sum of element-wise multiplication
function convolve(image, kernel, x, y) {
    let sum = 0;
    for (let i = 0; i < 3; i++) {
        for (let j = 0; j < 3; j++) {
            sum += image[x + i][y + j] * kernel[i][j];
        }
    }
    return sum;
}

// Convolution
let conv = [];
for (let i = 0; i < 3; i++) {
    let row = [];
    for (let j = 0; j < 3; j++) {
        row.push(convolve(image, kernel, i, j));
    }
    conv.push(row);
}
console.log("Convolution:\n", conv);

// ReLU
let relu = conv.map(row => row.map(value => Math.max(0, value)));
console.log("\nReLU:\n", relu);

// Max Pooling
let pool = [];
for (let i = 0; i < 2; i += 2) {
    let row = [];
    for (let j = 0; j < 2; j += 2) {
        let maxVal = Math.max(
            relu[i][j], relu[i][j + 1],
            relu[i + 1][j], relu[i + 1][j + 1]
        );
        row.push(maxVal);
    }
    pool.push(row);
}
console.log("\nMax Pooling:\n", pool);
