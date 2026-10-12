from pathlib import Path
from PIL import Image
import numpy as np


#Väljer bildfil, sparar som 3D matris
SCRIPT_DIR = Path(__file__).resolve().parent / "standard_test_images"

IMAGE_PATH = SCRIPT_DIR / "mandril_color.tif"
img = Image.open(IMAGE_PATH)
matrix = np.array(img)

#skriver ut matrisens storlek i samtliga dimensioner
print("Shape: ", matrix.shape)

#Omvandlar matrisen 3D -> 1D
flat_pixels = matrix.flatten()

#Skapar en 1D matris med slumpmässiga värden från 0 -> flat_pixels.size
perm_indices = np.random.permutation(flat_pixels.size)

#Kopierar värden från flat_pixels i slumpmässig ordning till ny 1D matris
shuffled_pixels = flat_pixels[perm_indices]

#Omvandlar matrisen 1D -> 3D 
reshaped_matrix = shuffled_pixels.reshape(matrix.shape)

#Öppnar original matrisen och den permuterade matrisen som en bild
img.show()
Image.fromarray(reshaped_matrix).show()

