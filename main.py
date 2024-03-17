# Import Python libraries
from PIL import Image

# Import structures from other files
from model import Model
from image import greyscale, black, resample, arrayed

# Print manifest
print("\033c", end="", flush=True)
print("Year 4 Mathematical Exploration 2024: Implementation", end="\033[K\n\a")
print("Janav Nagapatla and Aiden Lim", end="\033[K\n")
print("All rights reserved", end="\033[K\n")

# Import model
print("", end="\033[K\n")
print("Importing model", end="\033[K\n")

model: Model = Model(input("> Path to your model file (.networkconfig): \a\033[K"))

print(f"    > Model name: {model.name}", end="\033[K\n")
print(f"    > Model program: {model.program}", end="\033[K\n")
print(f"    > Model authors: {model.authors}", end="\033[K\n")
print(f"    > Model layers: {len(model.layers)}", end="\033[K\n")

print("> Successfully imported model", end="\033[K\n")

# Process image
print("", end="\033[K\n")
print("Processing image", end="\033[K\n")

image: Image = (
    resample(
        greyscale(
            black(
                greyscale(
                    Image.open(input("> Path to your image file (any type/resolution): \a\033[K")))))))

if input("> Would you like to view the image in Preview.app or MS Paint (yes/no): \a\033[K").lower()[0] == "y":
    image.show()

print("> Successfully imported image file", end="\033[K\n")

# Predicting result
print("", end="\033[K\n")
print("Predicting value of image", end="\033[K\n")

prediction, confidence = model.predict(arrayed(image))

print(f"> {model.name} predicts that the image is a {prediction} with {confidence:.2%} confidence", end="\033[K\n")
