import ImageToMatrix as itm
import numpy as np
from matt_enkel import entropi, korrelation, grannkorrelation, npcr_uaci

# Samma bilder som i ImageToMatrix.py

original = itm.matrix
krypterad = itm.chaoticencryptedArray.reshape(original.shape)
ratt_nycklar = itm.chaotickeys.T

#kopierar rätt nyckel men ändrar de första 3 elementen till 1. Borde kanske bara ändra enskilda bits?
fel_nycklar = ratt_nycklar.copy()
fel_nycklar[0,:] = [1,1,1]

# --- Dekryptering med rätt nycklar------------------------------------------------------------
# desubstituerad = np.bitwise_xor(krypterad.reshape(-1,3), ratt_nycklar)
# dekrypterad = itm.InverseArnoldCatMapTransform(desubstituerad.reshape(original.shape),10, 1, 1)

# --- Dekryptering med Near-Miss keys ------------------------------------------------------------
desubstituerad = np.bitwise_xor(krypterad.reshape(-1,3), fel_nycklar).reshape(original.shape)
dekrypterad = itm.InverseArnoldCatMapTransform(desubstituerad, 9, 1, 1)

# --- Mått ------------------------------------------------------------
print(fel_nycklar)
print(f"Entropi original:         {entropi(original):.4f}")
print(f"Entropi enkrypteringbild:   {entropi(krypterad):.4f}")
print(f"Entropi dekrypteringbild:   {entropi(dekrypterad):.4f}")
print(f"Korrelation orig/dekrypterad:  {korrelation(original, dekrypterad):.4f}")


print()
for namn, bild in [("original", original), ("dekrypterad bild", dekrypterad)]:
    gk = grannkorrelation(bild)
    print(
        f"Grannkorrelation {namn}: " + ", ".join(f"{k} {v:.3f}" for k, v in gk.items())
    )

npcr, uaci = npcr_uaci(original, dekrypterad)
print()
print(f"NPCR: {npcr:.2f} %   UACI: {uaci:.2f} %")
