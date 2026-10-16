import numpy as np

# 1. ENTROPI

# Formel:   H = - summa över alla värden v av  p(v) * log2( p(v) )
# där p(v) = andelen pixlar som har värdet v.
#
# Tolkning (för 8-bitars bild):
#   H = 0  -> alla pixlar har samma värde
#   H = 8  -> alla 256 värden är lika vanliga (perfekt brus)

def entropi(bild):
    # Steg 1: platta ut bilden till en lång lista med värden
    varden = bild.ravel()

    # Steg 2: räkna hur många pixlar som har värdet 0, 1, 2, ..., 255
    antal = np.bincount(varden, minlength=256)

    # Steg 3: gör om antal till andelar (sannolikheter). Summan blir 1.
    p = antal / varden.size

    # Steg 4: ta bort värden som aldrig förekommer (p = 0), eftersom
    # log2(0) inte finns. Termen p*log2(p) går mot 0 för p -> 0,
    # så det är korrekt att hoppa över dem.
    p = p[p > 0]

    # Steg 5: själva formeln
    H = -np.sum(p * np.log2(p))
    return float(H)


# ====================================================================
# 2. KORRELATION (Pearson) mellan två listor av samma längd
# ====================================================================
# Formel:   r = summa( (a_i - medel_a) * (b_i - medel_b) )
#               ---------------------------------------------
#               rot( summa (a_i-medel_a)^2 ) * rot( summa (b_i-medel_b)^2 )
#
# Tolkning:
#   r =  1  -> b följer a perfekt (samma bild, eventuellt ljusare/mörkare)
#   r =  0  -> ingen likhet (brus)
#   r = -1  -> b är en negativ version av a
def korrelation(a, b):
    # Steg 1: platta ut och gör om till decimaltal. Viktigt: med uint8
    # kan subtraktion "rulla runt" (3 - 5 = 254), så vi byter datatyp.
    a = a.ravel().astype(np.float64)
    b = b.ravel().astype(np.float64)

    # Steg 2: dra bort medelvärdet så att båda listorna centreras kring 0
    a_c = a - a.mean()
    b_c = b - b.mean()

    # Steg 3: täljaren = hur mycket a och b varierar åt samma håll
    taljare = np.sum(a_c * b_c)

    # Steg 4: nämnaren = hur mycket var och en varierar totalt
    namnare = np.sqrt(np.sum(a_c ** 2)) * np.sqrt(np.sum(b_c ** 2))

    # Steg 5: om en bild är helt enfärgad blir nämnaren 0 (division med
    # noll). Då finns ingen korrelation att mäta, och vi returnerar 0.
    if namnare == 0:
        return 0.0
    return float(taljare / namnare)


# ====================================================================
# 2b. KORRELATION MELLAN GRANNPIXLAR (ett mått på en enda bild)
# ====================================================================
# Idé: para ihop varje pixel med sin granne och räkna korrelation mellan
# "alla pixlar" och "alla deras grannar".
#   Naturlig bild: grannar liknar varandra -> nära 1
#   Bra kryptobild: grannar är oberoende  -> nära 0
def grannkorrelation(bild):
    # Gör om till decimaltal (samma skäl som ovan)
    b = bild.astype(np.float64)

    # Horisontellt: kolumn 0..N-2 mot kolumn 1..N-1 (granne åt höger)
    horisontellt = korrelation(b[:, :-1], b[:, 1:])
    # Vertikalt: rad 0..N-2 mot rad 1..N-1 (granne nedåt)
    vertikalt = korrelation(b[:-1, :], b[1:, :])
    # Diagonalt: granne snett nedåt åt höger
    diagonalt = korrelation(b[:-1, :-1], b[1:, 1:])

    return {"horisontellt": horisontellt,
            "vertikalt": vertikalt,
            "diagonalt": diagonalt}


# ====================================================================
# 3. NPCR och UACI mellan två bilder a och b av samma storlek
# ====================================================================
# NPCR = andel pixlar som har olika värde i a och b (i procent)
#        NPCR = 100 * (antal pixlar där a != b) / (totalt antal pixlar)
#
# UACI = genomsnittlig skillnad i värde, som procent av största värdet 255
#        UACI = 100 * medelvärdet av |a_i - b_i| / 255
#
# Förväntat för två OBEROENDE slumpbilder (8 bitar):
#   NPCR ≈ 99,61 %   (chansen att två slumpvärden är lika är 1/256)
#   UACI ≈ 33,46 %
# För exakta gränser, se Wu, Noonan & Agaian (2011); de beror på bildstorleken.
def npcr_uaci(a, b):
    # Gör om till heltal med tecken. Annars blir |a - b| fel med uint8
    # (t.ex. 3 - 5 blir 254 i stället för -2).
    a = a.astype(np.int64)
    b = b.astype(np.int64)

    # NPCR: jämför pixel för pixel; (a != b) ger True/False, medelvärdet
    # av det är andelen True.
    npcr = np.mean(a != b) * 100

    # UACI: absolutbelopp av skillnaden, dela med 255, ta medelvärdet.
    uaci = np.mean(np.abs(a - b) / 255.0) * 100

    return float(npcr), float(uaci)


# ====================================================================
# KONTROLL: stämmer mina handräknade svar med färdiga funktioner?
# ====================================================================
if __name__ == "__main__":
    from scipy.stats import entropy as scipy_entropi

    rng = np.random.default_rng(0)

    # Testbild: gradient + lite brus, så att pixlarna är ordnade men inte
    # identiska. Två bilder att jämföra.
    y, x = np.mgrid[0:100, 0:100]
    bild_a = np.clip(x * 1.5 + y + rng.normal(0, 5, (100, 100)), 0, 255)
    bild_a = bild_a.astype(np.uint8)
    bild_b = np.clip(bild_a + rng.normal(0, 20, (100, 100)), 0, 255)
    bild_b = bild_b.astype(np.uint8)

    print("ENTROPI")
    mitt = entropi(bild_a)
    sp = scipy_entropi(np.bincount(bild_a.ravel(), minlength=256), base=2)
    print(f"  mitt svar: {mitt:.6f}   scipy: {sp:.6f}\n")

    print("KORRELATION")
    mitt = korrelation(bild_a, bild_b)
    npk = np.corrcoef(bild_a.ravel(), bild_b.ravel())[0, 1]
    print(f"  mitt svar: {mitt:.6f}   numpy: {npk:.6f}\n")

    print("NPCR / UACI")
    npcr, uaci = npcr_uaci(bild_a, bild_b)
    npcr_k = np.sum(bild_a != bild_b) / bild_a.size * 100
    uaci_k = np.sum(np.abs(bild_a.astype(int) - bild_b.astype(int))) \
        / (255 * bild_a.size) * 100
    print(f"  NPCR  mitt: {npcr:.4f}   kontroll: {npcr_k:.4f}")
    print(f"  UACI  mitt: {uaci:.4f}   kontroll: {uaci_k:.4f}\n")

    print("TVÅ OBEROENDE SLUMPBILDER (ska ge ca 99,6 % och 33,4 %)")
    s1 = rng.integers(0, 256, (512, 512), dtype=np.uint8)
    s2 = rng.integers(0, 256, (512, 512), dtype=np.uint8)
    npcr, uaci = npcr_uaci(s1, s2)
    print(f"  NPCR {npcr:.2f} %   UACI {uaci:.2f} %")
    print(f"  Entropi slumpbild: {entropi(s1):.4f} (ska vara nära 8)")