from os import stat_result
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from PIL import Image
from scipy.integrate import solve_ivp


# Permuterar alla pixlar
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


# Permuterar alla pixlar
def InverseArnoldCatMapTransform(
    keystream: NDArray, iterations: int, Q: int, P: int
) -> NDArray:

    N, M = keystream.shape[0], keystream.shape[1]

    x, y = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")

    x_new = (x + P * y) % N
    y_new = (Q * x + (P * Q + 1) * y) % N

    result = keystream.copy()
    for _ in range(iterations):
        result = result[x_new, y_new]
    return result


# Skapar en kaotisk nyckel genom att iterera lorenz system
def lorenzsColorShuffle(
    number: int,
    x0: float,
    y0: float,
    z0: float,
    sigma: float,
    rho: float,
    beta: float,
    transient: float,
    dt: float,
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

# Använder en nyckel för att ändra pixel värden, använd denna eller XOR bitwise
def lorenzDiffuse(matrix: NDArray, keys: NDArray) -> NDArray:
    pixels = matrix.reshape(-1, 3).copy()
    ks = keys.reshape(-1, 3)

    out = np.zeros_like(pixels)
    prev = ks[0]

    for i in range(pixels.shape[0]):
        out[i] = (pixels[i].astype(int) + ks[i].astype(int) + prev.astype(int)) % 256
        prev = out[i]
    return out.reshape(matrix.shape)

def inverseLorenzDiffuse(matrix: NDArray, keys: NDArray) ->NDArray:
    pixels = matrix.reshape(-1,3).copy()
    ks = keys.reshape(-1, 3)

    out = np.zeros_like(pixels)
    prev = ks[0]

    for i in range(pixels.shape[0]):
        if i == 0: 
            prev = ks[0].astype(int) 
        else:
            prev = pixels[i-1]
        out[i] = (pixels[i].astype(int) - ks[i].astype(int) - prev.astype(int)) % 256
    return out.reshape(matrix.shape)


# Väljer bildfil, sparar som 3D matris
SCRIPT_DIR = Path(__file__).resolve().parent / "standard_test_images"

IMAGE_PATH = SCRIPT_DIR / "mandril_color.tif"
img = Image.open(IMAGE_PATH)
matrix = np.array(img)

# skriver ut matrisens storlek i samtliga dimensioner
print("Encrypting image with shape: ", matrix.shape)

chaosShuffledArray = ArnoldCatMapTransform(matrix, 10, 1, 1)
chaosShuffledImg = Image.fromarray(chaosShuffledArray)

# Sparar krypterad 3D matris som bild igen
H, W = matrix.shape[:2]
chaotickeys = lorenzsColorShuffle(
    H * W, 1.0, 0.0, 1.05, 10.0, 28.0, 8.0 / 3.0, 50.0, 0.005
)

chaoticencryptedArray = lorenzDiffuse(chaosShuffledArray, chaotickeys.T)
# chaoticencryptedArray = np.bitwise_xor(chaosShuffledArray.reshape(-1, 3), chaotickeys.T)
chaosEncryptedImg = Image.fromarray(chaoticencryptedArray.reshape(matrix.shape))

# Dekrypterar diffusionen genom samma bitwise XOR igen m.h.a nyckeln chaotickeys
# decrypted_flat = np.bitwise_xor(chaoticencryptedArray.reshape(-1, 3), chaotickeys.T)
decrypted_flat = inverseLorenzDiffuse(chaoticencryptedArray, chaotickeys.T)

# Sorterar tillbaka pixlarna till deras ursprungliga plats m.h.a nycklarna iterations, P och Q
inverse_perm = InverseArnoldCatMapTransform(decrypted_flat.reshape(matrix.shape), 10, 1, 1)

# Sparar dekrypterad 3D matris som bild igen
decrypted_img = Image.fromarray(inverse_perm)
'''
colors = ["red", "green", "blue"]
names = ["R", "G", "B"]


colors = ["red", "green", "blue"]
names = ["R", "G", "B"]

fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
for i, ax in enumerate(axes):
    counts, bins = np.histogram(matrix[:, :, i].ravel(), bins=256, range=(0, 256))
    ax.stairs(counts, bins, color=colors[i], fill=True, edgecolor=colors[i])
    ax.set_title(f"{names[i]} channel")
    ax.set_xlim(0, 255)
    ax.set_xlabel("Pixel value")
axes[0].set_ylabel("Count")
plt.tight_layout()
plt.show()
'''

# Plottar bilder med matplotlib och respektive gråskala histogram
def plotImages(bild1: NDArray, bild2: NDArray, bild3: NDArray):

    plt.subplot(2, 3, 1)
    plt.imshow(Image.fromarray(bild1))
    plt.axis("off")
    plt.title("Original Image")

    plt.subplot(2, 3, 4)
    plt.tick_params(axis='both', labelsize=4)
    counts, bins = np.histogram(bild1.flatten())
    plt.stairs(counts, bins)
    plt.ylabel("Frequency")

    plt.subplot(2, 3, 2)
    plt.imshow(Image.fromarray(bild2))
    plt.axis("off")
    plt.title("Chaotic Cipher")

    plt.subplot(2, 3, 5)
    plt.tick_params(axis='both', labelsize=4)
    counts, bins = np.histogram(bild2.flatten())
    plt.stairs(counts, bins)
    plt.xlabel("RGB value")

    plt.subplot(2, 3, 3)
    plt.imshow(Image.fromarray(bild3))
    plt.axis("off")
    plt.title("Decrypted Image")

    plt.subplot(2, 3, 6)
    plt.tick_params(axis='both', labelsize=4)
    counts, bins = np.histogram(bild3.flatten())
    plt.stairs(counts, bins)
    plt.show()

plotImages(matrix, chaoticencryptedArray, inverse_perm)