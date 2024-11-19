// Sample data
let X = [2, 4, 10, 12, 3, 20, 30, 11, 25];  // 1D points to cluster
let K = 2;  // Number of clusters

// Initialize centroids manually
let centroid1 = X[0];
let centroid2 = X[1];

let cluster1 = [];
let cluster2 = [];

for (let i = 0; i < 5; i++) {  // Limit the number of iterations to avoid complexity
    // Clear clusters at the beginning of each iteration
    cluster1 = [];
    cluster2 = [];

    // Assign each point to the nearest centroid
    for (let point of X) {
        if (Math.abs(point - centroid1) < Math.abs(point - centroid2)) {
            cluster1.push(point);
        } else {
            cluster2.push(point);
        }
    }

    // Update centroids to the mean of each cluster
    if (cluster1.length > 0) {
        centroid1 = cluster1.reduce((sum, val) => sum + val, 0) / cluster1.length;
    }
    if (cluster2.length > 0) {
        centroid2 = cluster2.reduce((sum, val) => sum + val, 0) / cluster2.length;
    }
}

// Print final clusters and centroids
console.log("Cluster 1:", cluster1);
console.log("Cluster 2:", cluster2);
console.log("Final Centroid 1:", centroid1);
console.log("Final Centroid 2:", centroid2);
