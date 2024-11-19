import numpy as np


class SimpleLSTM:
    def __init__(self):
        # Initialize weights and biases as arrays for compactness
        self.W = np.array(
            [1.2, 1.1, 0.8, 1.0]
        )  # Forget, Input, Candidate, Output weights
        self.b = np.array(
            [-0.5, 0.0, 0.1, -0.3]
        )  # Forget, Input, Candidate, Output biases

    def step(self, x, h, C):
        # Perform all gate calculations
        f, i, C_tilde, o = self.sigmoid(self.W * (h + x) + self.b)

        # Update cell state and hidden state
        C = f * C + i * np.tanh(C_tilde)
        h = o * np.tanh(C)

        return h, C

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))


# Initialize and test
lstm = SimpleLSTM()
h, C = lstm.step(0.5, 0.0, 0.0)
print("Hidden State:", h, "Cell State:", C)
