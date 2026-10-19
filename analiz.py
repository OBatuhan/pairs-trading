"""
analiz.py — Projenin istatistik motoru.

Arayüzden (Streamlit'ten) bağımsızdır: Colab'a kopyalayıp aynen kullanabilirsin.
İçindekiler: keşif penceresi, korelasyon, Engle-Granger, half-life, z-skoru,
Benjamini-Hochberg düzeltmesi ve sinyal noktaları.
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests
from statsmodels.tsa.stattools import coint

ISLEM_GUNU_YIL = 252  # bir yılda yaklaşık işlem günü


def kesif_donemi(fiyatlar, yil, ay_geri=0):
    """
    Keşif penceresini keser: içinde bulunduğumuz aydan ÖNCEKİ son `yil` yıl.
    Örnek: bugün 14 Ekim 2026, yil=2 → 1 Ekim 2024 – 30 Eylül 2026.
    ay_geri=1 verilirse pencere bir ay geri kayar (= "geçen ayın keşfi").

    Bu ayın verisi keşifte kullanılmadığı için, ay içindeki z-skorları
    keşiften sonra gelen (örneklem dışı) veriyle hesaplanmış olur.
    """
    son_gun = fiyatlar.index[-1]
    bitis = son_gun.to_period("M").to_timestamp() - pd.DateOffset(months=ay_geri)
    baslangic = bitis - pd.DateOffset(years=yil)
    donem = fiyatlar.loc[(fiyatlar.index >= baslangic) & (fiyatlar.index < bitis)]
    return donem, bitis


def korelasyon_matrisi(fiyatlar):
    """Günlük GETİRİLER üzerinden korelasyon (fiyatlarla değil: sahte korelasyon tuzağı)."""
    getiriler = fiyatlar.pct_change(fill_method=None)
    return getiriler.corr()


def engle_granger(log_y, log_x):
    """
    Engle-Granger'ın iki adımı:
      1) OLS: log_y = alfa + beta * log_x + hata  → beta = hedge oranı, hata = spread
      2) Spread'e birim kök testi → coint() bunu doğru kritik değerlerle yapıp p-değerini verir.
    """
    model = sm.OLS(log_y, sm.add_constant(log_x)).fit()
    alfa = model.params.iloc[0]
    beta = model.params.iloc[1]
    spread = model.resid
    p_degeri = coint(log_y, log_x)[1]
    return alfa, beta, p_degeri, spread


def half_life(spread):
    """
    Spread'in ortalamaya yarı yolda dönmesi kaç gün sürer?
    Regresyon: Δspread(t) = c + λ · spread(t−1)  →  half-life = −ln(2) / λ
    λ ≥ 0 ise ortalamaya dönüş yok demektir → sonsuz.
    """
    s = spread.dropna().values
    gecikmeli = s[:-1]
    fark = np.diff(s)
    model = sm.OLS(fark, sm.add_constant(gecikmeli)).fit()
    lam = model.params[1]
    if lam >= 0:
        return np.inf
    return -np.log(2) / lam


def z_skoru(spread, pencere):
    """Makasın son `pencere` günün ortalamasından kaç standart sapma uzakta olduğu."""
    ortalama = spread.rolling(pencere).mean()
    std = spread.rolling(pencere).std()
    return (spread - ortalama) / std


def ciftleri_tara(fiyatlar, ciftler, kesif_yil, kor_esik, p_esik, ay_geri=0):
    """
    Huninin tamamı: olası çiftler → korele olanlar → eşbütünleşik olanlar.

    Dönenler:
      sonuc       : korelasyon eşiğini geçen her çift için bir satır (p-değerleri, beta, half-life...)
      huni        : her aşamada kaç çift kaldığı
      kesif_tarihi: keşfin yapıldığı ay başı
    """
    donem, kesif_tarihi = kesif_donemi(fiyatlar, kesif_yil, ay_geri)
    korelasyonlar = korelasyon_matrisi(donem)
    min_gun = int(0.8 * ISLEM_GUNU_YIL * kesif_yil)  # penceredeki günlerin en az %80'i dolu olsun

    satirlar = []
    korele_sayisi = 0
    for y, x in ciftler:
        if y not in donem.columns or x not in donem.columns:
            continue
        ikili = donem[[y, x]].dropna()
        if len(ikili) < min_gun:
            continue

        # 1. eleme: korelasyon (hızlı)
        r = korelasyonlar.loc[y, x]
        if pd.isna(r) or r < kor_esik:
            continue
        korele_sayisi += 1

        # 2. eleme: Engle-Granger (yavaş, o yüzden sadece korele çiftlere)
        log_y = np.log(ikili[y])
        log_x = np.log(ikili[x])
        alfa, beta, p_degeri, spread = engle_granger(log_y, log_x)

        satirlar.append({
            "cift": f"{y} / {x}",
            "y": y,
            "x": x,
            "korelasyon": r,
            "p": p_degeri,
            "p_bh": np.nan,
            "alfa": alfa,
            "beta": beta,
            "half_life": half_life(spread),
        })

    sutunlar = ["cift", "y", "x", "korelasyon", "p", "p_bh", "alfa", "beta", "half_life"]
    sonuc = pd.DataFrame(satirlar, columns=sutunlar)

    # Çoklu test düzeltmesi: bu ay yapılan tüm Engle-Granger testleri birlikte düzeltilir.
    if len(sonuc) > 0:
        sonuc["p_bh"] = multipletests(sonuc["p"], method="fdr_bh")[1]

    huni = {
        "olasi": len(ciftler),
        "korele": korele_sayisi,
        "esbutunlesik": int((sonuc["p"] < p_esik).sum()),
        "bh": int((sonuc["p_bh"] < p_esik).sum()),
    }
    return sonuc, huni, kesif_tarihi


def spread_ve_z(fiyatlar, y, x, alfa, beta, z_pencere):
    """
    Keşifte bulunan alfa ve beta ile spread'i TÜM tarih için hesaplar, sonra kayan z-skoru.
    Keşif tarihinden sonrası örneklem dışıdır (beta o veriyi görmedi).
    """
    ikili = fiyatlar[[y, x]].dropna()
    spread = np.log(ikili[y]) - beta * np.log(ikili[x]) - alfa
    return spread, z_skoru(spread, z_pencere)


def sinyal_noktalari(z, esik):
    """|z|'nin eşiği dışarı doğru ilk geçtiği günler (sinyalin doğduğu anlar)."""
    disarida = z.abs() > esik
    giris = disarida & ~disarida.shift(1, fill_value=False)
    return z[giris]


def durum_etiketi(z, esik):
    """Tablodaki 'Durum' sütunu için kısa etiket. Isınma eşiği = z eşiğinin %75'i (2 → 1.5)."""
    if pd.isna(z):
        return "—"
    if abs(z) > esik:
        return "🔔 Sinyal"
    if abs(z) > 0.75 * esik:
        return "🌡️ Isınıyor"
    return "Sakin"
