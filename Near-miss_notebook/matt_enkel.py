import numpy as np

# 1. ENTROPI

def entropi(bild):
    # Bild blir lista
    varden = bild.ravel()

    # Räkna förekomst av varje värde 0-255
    antal = np.bincount(varden, minlength=256)

    # Antal blir andelar
    p = antal / varden.size

    # Ta bort värden där p = 0, eftersom log2(0) inte finns
    p = p[p > 0]

    # Entropiformeln
    H = -np.sum(p * np.log2(p))
    return float(H)

# 2. KORRELATION 

def korrelation(a, b):
    # Gör till lista.
    # Gör om till decimaltal
    # uint8 jobbar mod, så vi byter datatyp

    a = a.ravel().astype(np.float64)
    b = b.ravel().astype(np.float64)

    # Dra bort medelvärdet
    # båda listorna centreras kring 0
    a_c = a - a.mean()
    b_c = b - b.mean()

    # Täljaren 
    taljare = np.sum(a_c * b_c)

    # Nämnaren 
    namnare = np.sqrt(np.sum(a_c ** 2)) * np.sqrt(np.sum(b_c ** 2))

    # Om enfärgad blir nämnaren 0.
    # returnera 0
    if namnare == 0:
        return 0.0
    return float(taljare / namnare)

# 2b. KORRELATION MELLAN GRANNPIXLAR
# Använd grannkorrelation(bild) för att få korrelationen mellan grannpixlar
# : horisontellt, vertikalt och diagonalt.

def grannkorrelation(bild):
    b = bild.astype(np.float64)

    horisontellt = korrelation(b[:, :-1], b[:, 1:])
    vertikalt = korrelation(b[:-1, :], b[1:, :])
    diagonalt = korrelation(b[:-1, :-1], b[1:, 1:])

    return {"horisontellt": horisontellt,
            "vertikalt": vertikalt,
            "diagonalt": diagonalt}


# 3. NPCR och UACI

def npcr_uaci(a, b):

    # Gör om till int64 för att undvika overflow
    a = a.astype(np.int64)
    b = b.astype(np.int64)

    # NPCR: jämför pixel för pixel
    # (a != b) ger True/False, medelvärdet
    # andelen True
    npcr = np.mean(a != b) * 100

    # UACI: absolutbelopp av skillnaden
    # dela med 255
    # ta medelvärdet.
    uaci = np.mean(np.abs(a - b) / 255.0) * 100

    return float(npcr), float(uaci)