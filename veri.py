"""
veri.py — Yahoo Finance'ten fiyat verisini çeken dosya.

Önbellek (cache) app.py içinde yapılıyor: veri bir kez indirilir,
6 saat boyunca tekrar indirilmeden kullanılır.
"""

import pandas as pd
import yfinance as yf


def fiyatlari_indir(semboller, period="10y"):
    """
    Verilen hisselerin günlük kapanış fiyatlarını indirir.
    auto_adjust=True: fiyatlar temettü ve hisse bölünmelerine göre düzeltilir
    (yoksa bölünme günlerinde sahte "çöküşler" görürüz).

    Dönen tablo: satırlar = tarihler, sütunlar = hisse sembolleri.
    """
    ham = yf.download(
        list(semboller),
        period=period,
        auto_adjust=True,
        progress=False,
        threads=True,
    )

    if ham is None or ham.empty:
        raise RuntimeError(
            "Yahoo Finance'ten veri gelmedi. Biraz bekleyip sayfayı yenilemeyi dene."
        )

    # Birden çok hisse indirince sütunlar iki katmanlı olur: ("Close", "V"), ("Close", "MA")...
    # Biz sadece kapanış (Close) katmanını alıyoruz.
    if isinstance(ham.columns, pd.MultiIndex):
        fiyatlar = ham["Close"]
    else:
        fiyatlar = ham[["Close"]].rename(columns={"Close": list(semboller)[0]})

    # Hiç verisi gelmeyen hisseleri at
    fiyatlar = fiyatlar.dropna(axis=1, how="all")

    # Tarihlerde saat dilimi bilgisi varsa kaldır (tarih karşılaştırmaları sorunsuz olsun)
    if fiyatlar.index.tz is not None:
        fiyatlar.index = fiyatlar.index.tz_localize(None)

    return fiyatlar.sort_index()
