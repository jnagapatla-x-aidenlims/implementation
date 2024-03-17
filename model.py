# Import Python libraries
import numpy as np

# Import structures from other files
from layer import Layer
from helpers import sigmoid, tanh, softmax


class Model:
    """
    Using a model file from a trainer, one can predict images using this class
    """

    def __init__(self,
                 model_path: str) -> None:
        """
        Imports the weights of a trained model into the program
        """

        self.layers: list[Layer] = []

        with open(model_path, "r") as model_file:
            self.name: str = model_file.readline()[7:-1]
            self.program: str = model_file.readline()[9:-1]
            self.authors: str = model_file.readline()[9:-1]
            model_file.readline()

            while True:
                match model_file.readline():
                    case "> New Layer\n":
                        input_size: int = int(model_file.readline()[22:-1])
                        output_size: int = int(model_file.readline()[23:-1])
                        activation: str = model_file.readline()[27:-1]
                        model_file.readline()
                        weights: np.ndarray = np.array([[float(model_file.readline()[10:-1]) for _ in range(input_size)]
                                                        for _ in range(output_size)])
                        model_file.readline()
                        biases: np.ndarray = np.array([[float(model_file.readline()[10:-1])] for _ in range(output_size)])

                        match activation:
                            case "sigmoid":
                                self.layers.append(Layer(weights, biases, sigmoid))
                            case "tanh":
                                self.layers.append(Layer(weights, biases, tanh))
                    case "--- End Network Configuration ---":
                        break

    def predict(self,
                image: np.ndarray) -> tuple[int, float]:
        """
        Feeds the image array through the program
        """

        evaluation: np.ndarray = image

        for layer in self.layers[0:-1]:
            evaluation = layer.forward(evaluation)

        probability = softmax(self.layers[-1].nonactivated(evaluation))

        evaluation = self.layers[-1].forward(evaluation)

        return int(np.argmax(evaluation)), probability


if __name__ == "__main__":
    exit("This script cannot be run on its own.\033[K")
