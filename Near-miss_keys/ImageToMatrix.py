from pathlib import Path
from PIL import Image
import numpy as np



SCRIPT_DIR = Path(__file__).resolve().parent / "standard_test_images"

IMAGE_PATH = SCRIPT_DIR / "mandril_color.tif"
img = Image.open(IMAGE_PATH)
matrix = np.array(img)
print("test")

print("Shape: ", matrix)

flat_pixels = matrix.flatten()
# print(flat_pixels)

perm_indices = np.random.permutation(matrix.size)
shuffled_pixels = flat_pixels[perm_indices]

# print (shuffled_pixels)

# shufflePixels(matrix)
reshaped_matrix = shuffled_pixels.reshape(matrix.shape)
img.show()
Image.fromarray(reshaped_matrix).show()


    ##img.show()
""" def shufflePixels(matrix):
    shuffledMatrix = np.array(matrix.shape)
    print("Shape: ", shuffledMatrix.shape)
 """


