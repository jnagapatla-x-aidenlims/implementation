# Import Python libraries
import numpy as np
from typing import Callable


class Layer:
    """
    A representation of a connected layer
    Requires the number of input and output neurones of the layer and its desired activation function + derivative
    """

    def __init__(self,
                 weights: np.ndarray,
                 biases: np.ndarray,
                 activation: Callable[[np.ndarray], np.ndarray]) -> None:
        """
        Generates random weights and biases (as a starting point) or the layer
        """

        self.weights: np.ndarray = weights
        self.biases: np.ndarray = biases

        self.activation: Callable[[np.ndarray], np.ndarray] = activation

    def forward(self,
                previous: np.ndarray) -> np.ndarray:
        """
        Conducts forward propagation and returns the output neurones
        """

        return self.activation(np.dot(self.weights, previous) + self.biases)

    def nonactivated(self,
                     previous: np.ndarray) -> np.ndarray:
        """
        Conducts forward propagation without activation and returns the output neurones
        """

        return np.dot(self.weights, previous) + self.biases


if __name__ == "__main__":
    exit("This script cannot be run on its own.\033[K")
