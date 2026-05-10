class Solution:
    def get_minimizer(self, iterations: int, learning_rate: float, init: int) -> float:
        # Objective function: f(x) = x^2
        # Derivative:         f'(x) = 2x
        # Update rule:        x = x - learning_rate * f'(x)
        # Round final answer to 5 decimal places
        current_value = init 
        for i in range(iterations):
            current_value = current_value - learning_rate * (2 * current_value)
        return round(current_value, 5)