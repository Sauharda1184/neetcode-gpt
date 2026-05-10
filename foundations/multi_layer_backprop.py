import numpy as np
from typing import List


class Solution:
    def forward_and_backward(self,
                              x: List[float],
                              W1: List[List[float]], b1: List[float],
                              W2: List[List[float]], b2: List[float],
                              y_true: List[float]) -> dict:
        # Convert inputs to numpy arrays
        x = np.array(x)
        W1 = np.array(W1)
        b1 = np.array(b1)
        W2 = np.array(W2)
        b2 = np.array(b2)
        y_true = np.array(y_true)

        # Architecture: x -> Linear(W1, b1) -> ReLU -> Linear(W2, b2) -> predictions
        # Loss: MSE = mean((predictions - y_true)^2)
        
        # Note: z = xW^T + b logic used here to match the architecture description
        z1 = x @ W1.T + b1

        a1 = np.maximum(0, z1)

        predictions = a1 @ W2.T + b2 

        loss = np.mean((predictions - y_true)**2)

        n = len(y_true)

        d_predictions = (2 / n) * (predictions - y_true)

        # Gradient w.r.t W2: (out_size, 1) @ (1, hidden_size)
        dW2 = np.outer(d_predictions, a1)

        db2 = d_predictions

        # Backprop through W2
        da1 = d_predictions @ W2

        dz1 = da1 * (z1 > 0)

        # Gradient w.r.t W1: (hidden_size, 1) @ (1, input_size)
        dW1 = np.outer(dz1, x)

        db1 = dz1

        return {
            "loss": round(float(loss), 4),
            "dW1": (np.round(dW1, 4) + 0.0).tolist(),
            "db1": (np.round(db1, 4) + 0.0).tolist(),
            "dW2": (np.round(dW2, 4) + 0.0).tolist(),
            "db2": (np.round(db2, 4) + 0.0).tolist()
        }
