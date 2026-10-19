"""
hisseler.py — Hangi hisseleri tarayacağımızı belirleyen dosya.

Şimdilik S&P 500'den 3 sektör, her birinde 10 hisse (toplam 30).
İleride S&P 500'ün tamamına geçmek için sadece bu dosyayı değiştirmek yeterli olacak;
diğer dosyalar hisse listesini buradan okuyor.
"""

from itertools import combinations

# Sözlük yapısı: {sektör adı: {hisse sembolü: şirket adı}}
SEKTORLER = {
    "Finans & Ödeme": {
        "V": "Visa",
        "MA": "Mastercard",
        "AXP": "American Express",
        "PYPL": "PayPal",
        "JPM": "JPMorgan Chase",
        "BAC": "Bank of America",
        "C": "Citigroup",
        "WFC": "Wells Fargo",
        "GS": "Goldman Sachs",
        "MS": "Morgan Stanley",
    },
    "Teknoloji": {
        "AAPL": "Apple",
        "MSFT": "Microsoft",
        "GOOGL": "Alphabet (Google)",
        "META": "Meta",
        "NVDA": "Nvidia",
        "AMD": "AMD",
        "INTC": "Intel",
        "AVGO": "Broadcom",
        "QCOM": "Qualcomm",
        "TXN": "Texas Instruments",
    },
    "Enerji": {
        "XOM": "ExxonMobil",
        "CVX": "Chevron",
        "COP": "ConocoPhillips",
        "EOG": "EOG Resources",
        "SLB": "SLB (Schlumberger)",
        "HAL": "Halliburton",
        "OXY": "Occidental",
        "PSX": "Phillips 66",
        "MPC": "Marathon Petroleum",
        "VLO": "Valero",
    },
}


def semboller(sektorler):
    """Seçili sektörlerdeki tüm hisse sembollerini tek bir liste olarak döndürür."""
    liste = []
    for sektor in sektorler:
        liste.extend(SEKTORLER[sektor].keys())
    return liste


def sektor_ici_ciftler(sektorler):
    """
    Her sektörün KENDİ İÇİNDEKİ tüm ikili kombinasyonları üretir.
    10 hisseli bir sektörde 10*9/2 = 45 çift olur; 3 sektörde 135 çift.
    Çiftin ilk elemanı regresyonda bağımlı değişken (y), ikincisi bağımsız (x).
    """
    ciftler = []
    for sektor in sektorler:
        ciftler.extend(combinations(SEKTORLER[sektor].keys(), 2))
    return ciftler


def sirket_adi(sembol):
    """Sembolden şirket adını bulur (ör. 'V' -> 'Visa')."""
    for hisseler in SEKTORLER.values():
        if sembol in hisseler:
            return hisseler[sembol]
    return sembol
