from os import stat_result
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from PIL import Image
from scipy.integrate import solve_ivp

#Permuterar alla pixlar
def ArnoldCatMapTransform(matrix: NDArray, iterations: int, Q: int, P: int) -> NDArray:

    N, M = matrix.shape[0], matrix.shape[1]

    x, y = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")

    x_new = (x + P * y) % N
    y_new = (Q * x + (P * Q + 1) * y) % N

    result = matrix.copy()
    for _ in range(iterations):
        transformed = np.empty_like(result)
        transformed[x_new, y_new] = result
        result = transformed

    return result

#Skapar en kaotisk nyckel genom att iterera lorenz system 
def lorenzsColorShuffle(number: int, x0: float, y0: float, z0: float,sigma: float, rho: float, beta: float, transient: float, dt: float,
) -> NDArray:
    def lorenz_deriv(t, state, s, r, b):
        x, y, z = state
        return [sigma * (y - x), x * (rho - z) - y, x * y - beta * z]

    t_eval = transient + dt * np.arange(number)

    sol = solve_ivp(
        lorenz_deriv,
        (0, t_eval[-1]),
        [x0, y0, z0],
        args=(10.0, 28.0, 8.0 / 3.0),
        t_eval=t_eval,
        method="RK45",
        rtol=1e-6,
        atol=1e-6,
    )
    keys = (np.abs(sol.y) * 1e6).astype(np.uint64) % 256
    return keys.astype(np.uint8)

#Använder en nyckel för att ändra pixel värden, använd denna eller XOR bitwise
def lorenzDiffuse(matrix: NDArray, keys: NDArray) -> NDArray:
    pixels = matrix.reshape(-1, 3)
    print(ks.shape)
    ks = keys.T
    print(ks.shape)

    out = np.empty_like(pixels)
    prev = ks[0].copy()
    for i in range(pixels.shape[0]):
        out[i] = pixels[i] + ks[i] + prev
        prev = out[i]
    return out.reshape(matrix.shape)


# Väljer bildfil, sparar som 3D matris
SCRIPT_DIR = Path(__file__).resolve().parent / "standard_test_images"

IMAGE_PATH = SCRIPT_DIR / "mandril_color.tif"
img = Image.open(IMAGE_PATH)
matrix = np.array(img)

# skriver ut matrisens storlek i samtliga dimensioner
print("Encrypting image with shape: ", matrix.shape)

# Omvandlar matrisen 3D -> 1D
flat_pixels = matrix.flatten()

# Skapar en 1D matris med slumpmässiga värden från 0 -> flat_pixels.size
perm_indices = np.random.permutation(flat_pixels.size)

# Kopierar värden från flat_pixels i slumpmässig ordning till ny 1D matris (Permutation)
shuffled_pixels = flat_pixels[perm_indices]

chaosShuffledArray = ArnoldCatMapTransform(matrix, 10, 1, 1)
chaosShuffledImg = Image.fromarray(chaosShuffledArray)

# Ändrar på pixelvärden genom bitwise XOR med slumpmässig keystream (Diffusion)
keystream = np.random.randint(0, 256, size=flat_pixels.size, dtype=np.uint8)
encrypted_flat = np.bitwise_xor(shuffled_pixels, keystream)

# Omvandlar matrisen 1D -> 3D
reshaped_matrix = encrypted_flat.reshape(matrix.shape)

# Sparar krypterad 3D matris som bild igen
encrypted_img = Image.fromarray(reshaped_matrix)
H, W = matrix.shape[:2]
chaotickeystream = lorenzsColorShuffle(
    H * W, 1.0, 1.0, 1.0, 10.0, 28.0, 8.0 / 3.0, 50.0, 0.005
)
# chaoticencryptedArray = lorenzDiffuse(chaosShuffledArray, chaotickeystream)
chaoticencryptedArray = np.bitwise_xor(
    chaosShuffledArray.reshape(-1, 3), chaotickeystream.T
)
chaosEncryptedImg = Image.fromarray(chaoticencryptedArray.reshape(matrix.shape))
""" 
# Dekrypterar diffusionen genom samma bitwise XOR igen
decrypted_flat = np.bitwise_xor(chaoticencryptedArray.flatten(), keystream)

# Sorterar tillbaka pixlarna till dess ursprungliga plats m.h.a perm_indices
inverse_perm = np.argsort(perm_indices)
recovered_flat = decrypted_flat[inverse_perm]

# Omvandlar matrisen 1D -> 3D
decrypted_reshaped_matrix = recovered_flat.reshape(matrix.shape)

# Sparar dekrypterad 3D matris som bild igen
decrypted_img = Image.fromarray(decrypted_reshaped_matrix)
 """
# Öppnar original matrisen, permuterade matrisen och den dekryptedade matrisen som en bild
# img.show( title = 'Original bild')
# encrypted_img.show( title = 'krypterad bild')
# decrypted_img.show( title = 'dekrypterad bild')
""" 
plt.subplot(2, 3, 1)
plt.imshow(img)
plt.axis("off")
plt.title("Original Image")


plt.subplot(2, 3, 4)
counts, bins = np.histogram(matrix.flatten())
plt.stairs(counts, bins)

plt.subplot(2, 3, 2)
plt.imshow(chaosShuffledImg)
plt.axis("off")
plt.title("Chaotic Permutation")

plt.subplot(2, 3, 5)
counts, bins = np.histogram(chaosShuffledArray.flatten())
plt.stairs(counts, bins)

plt.subplot(2, 3, 3)
plt.imshow(chaosEncryptedImg)
plt.axis("off")
plt.title("Chaotic Cipher")

plt.subplot(2, 3, 6)
counts, bins = np.histogram(chaoticencryptedArray.flatten())
plt.stairs(counts, bins)
plt.show()
 """