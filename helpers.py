# Import Python libraries
import numpy as np


def sigmoid(value: np.ndarray) -> np.ndarray:
    """
    Returns logistic sigmoid at value
    """

    return 1.0 / (1.0 + np.exp(-value))


def tanh(value: np.ndarray) -> np.ndarray:
    """
    Returns tanh at value
    """

    return np.tanh(value)


def softmax(value: np.ndarray) -> float:
    """
    Returns softmax at value
    """

    return np.exp(value.max()) / np.sum(np.exp(value))


if __name__ == "__main__":
    exit("This script cannot be run on its own.\033[K")
