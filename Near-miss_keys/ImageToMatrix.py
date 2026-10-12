from pathlib import Path
from PIL import Image
import numpy as np
# import matplotlib.pyplot as plt


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

#Kopierar värden från flat_pixels i slumpmässig ordning till ny 1D matris (Permutation)
shuffled_pixels = flat_pixels[perm_indices]

#Ändrar på pixelvärden genom bitwise XOR med slumpmässig keystream (Diffusion)
keystream = np.random.randint(0,256, size=flat_pixels.size, dtype=np.uint8)
encrypted_flat = np.bitwise_xor(shuffled_pixels, keystream)

#Omvandlar matrisen 1D -> 3D 
reshaped_matrix = encrypted_flat.reshape(matrix.shape)

#Sparar krypterad 3D matris som bild igen
encrypted_img = Image.fromarray(reshaped_matrix)

#Dekrypterar diffusionen genom samma bitwise XOR igen
decrypted_flat = np.bitwise_xor(encrypted_flat, keystream)

#Sorterar tillbaka pixlarna till dess ursprungliga plats m.h.a perm_indices
inverse_perm = np.argsort(perm_indices)
recovered_flat = decrypted_flat[inverse_perm]

#Omvandlar matrisen 1D -> 3D 
decrypted_reshaped_matrix = recovered_flat.reshape(matrix.shape)

#Sparar dekrypterad 3D matris som bild igen
decrypted_img = Image.fromarray(decrypted_reshaped_matrix)

#Öppnar original matrisen, permuterade matrisen och den dekryptedade matrisen som en bild
img.show( title = 'Original bild')
encrypted_img.show( title = 'krypterad bild')
decrypted_img.show( title = 'dekrypterad bild')




