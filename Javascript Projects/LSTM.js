class SimpleLSTM {
    constructor() {
        // Initialize weights and biases as arrays for compactness
        this.W = [1.2, 1.1, 0.8, 1.0];  // Forget, Input, Candidate, Output weights
        this.b = [-0.5, 0.0, 0.1, -0.3]; // Forget, Input, Candidate, Output biases
    }

    step(x, h, C) {
        // Perform all gate calculations
        const f = SimpleLSTM.sigmoid(this.W[0] * (h + x) + this.b[0]);
        const i = SimpleLSTM.sigmoid(this.W[1] * (h + x) + this.b[1]);
        const C_tilde = SimpleLSTM.sigmoid(this.W[2] * (h + x) + this.b[2]);
        const o = SimpleLSTM.sigmoid(this.W[3] * (h + x) + this.b[3]);

        // Update cell state and hidden state
        C = f * C + i * Math.tanh(C_tilde);
        h = o * Math.tanh(C);

        return { h, C };
    }

    static sigmoid(x) {
        return 1 / (1 + Math.exp(-x));
    }
}

// Initialize and test
const lstm = new SimpleLSTM();
const result = lstm.step(0.5, 0.0, 0.0);
console.log("Hidden State:", result.h, "Cell State:", result.C);
