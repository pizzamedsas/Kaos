from ImageToMatrix import matrix, chaosShuffledArray, chaoticencryptedArray
from matt_enkel import entropi, korrelation, grannkorrelation, npcr_uaci

# Samma bilder som i ImageToMatrix.py
original = matrix
jämförelse05 = chaosShuffledArray.reshape(matrix.shape)
jämförelse = chaoticencryptedArray.reshape(matrix.shape)

# --- Mått ------------------------------------------------------------
print()
print(f"Entropi original:         {entropi(original):.4f}")
print(f"Entropi jämförelsebild:   {entropi(jämförelse):.4f}")
print(f"Korrelation orig/jämför:  {korrelation(original, jämförelse):.4f}")

print()
for namn, bild in [("original", original), ("jämförelsebild", jämförelse)]:
    gk = grannkorrelation(bild)
    print(
        f"Grannkorrelation {namn}: " + ", ".join(f"{k} {v:.3f}" for k, v in gk.items())
    )

npcr, uaci = npcr_uaci(original, jämförelse)
print()
print(f"NPCR: {npcr:.2f} %   UACI: {uaci:.2f} %")
