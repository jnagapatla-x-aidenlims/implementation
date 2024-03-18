# Import Python libraries
import numpy as np
from PIL import Image


def threshold(x: int) -> int:
    """
    Returns 255 if x is more than or equal to the cutoff (else 0)
    """

    return 255 if x >= 75 else 0


def greyscale(image: Image) -> Image:
    """
    Makes the image greyscale from coloured
    """

    return image.convert('L')


def black(image: Image) -> Image:
    """
    Makes the image black and white from coloured
    """

    return image.point(threshold, mode='1')


def resample(image: Image) -> Image:
    """
    Resizes and resamples via LANCZOS the image
    """

    result = Image.new("L", (28, 28), 255)
    result.paste(image.resize((20, 20), Image.Resampling.BICUBIC), (4, 4))
    return result


def arrayed(image: Image) -> np.ndarray:
    """
    Converts the image into input neurones
    """

    return 1 - np.array(image).reshape((784, 1)) / 255


if __name__ == "__main__":
    exit("This script cannot be run on its own.\033[K")
