"""
app.py — Uygulamanın arayüzü (Streamlit).

Akış: ayarlar (yan menü) → veriyi indir → çiftleri tara (bu ay + geçen ay)
→ huni sayacı → aktif / ısınan sinyaller → izleme listesi → seçilen çiftin detayı.
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from analiz import (
    ciftleri_tara,
    durum_etiketi,
    kesif_donemi,
    korelasyon_matrisi,
    sinyal_noktalari,
    spread_ve_z,
)
from hisseler import SEKTORLER, sektor_ici_ciftler, semboller, sirket_adi
from veri import fiyatlari_indir

st.set_page_config(page_title="Pairs Trading Tarayıcı", page_icon="📈", layout="wide")


# ---------------------------------------------------------------------------
# Önbellekli fonksiyonlar: aynı girdiyle tekrar çağrılınca hesaplamayı tekrarlamaz
# ---------------------------------------------------------------------------
@st.cache_data(ttl=6 * 60 * 60, show_spinner=False)  # veri 6 saatte bir tazelenir
def veri_yukle(sembol_listesi):
    return fiyatlari_indir(list(sembol_listesi))


@st.cache_data(show_spinner=False)
def tara(fiyatlar, ciftler, kesif_yil, kor_esik, p_esik, ay_geri):
    return ciftleri_tara(fiyatlar, list(ciftler), kesif_yil, kor_esik, p_esik, ay_geri)


def yorum(satir):
    """z > 0: spread ortalamanın üstünde → y, x'e göre görece pahalı."""
    if satir["z"] > 0:
        pahali, ucuz = satir["y"], satir["x"]
    else:
        pahali, ucuz = satir["x"], satir["y"]
    return f"{sirket_adi(pahali)} ({pahali}) görece pahalı, {sirket_adi(ucuz)} ({ucuz}) görece ucuz"


def kesif_cizgisi(fig, tarih):
    """Grafiğe 'keşif sonu' dikey çizgisi ekler: solu örneklem içi, sağı örneklem dışı."""
    gun = tarih.strftime("%Y-%m-%d")
    fig.add_vline(x=gun, line_dash="dot", line_color="gray")
    fig.add_annotation(x=gun, y=1, yref="paper", text="keşif sonu", showarrow=False,
                       xanchor="left", yanchor="bottom")


# ---------------------------------------------------------------------------
# Yan menü: ayarlar
# ---------------------------------------------------------------------------
st.sidebar.header("⚙️ Ayarlar")
secili_sektorler = st.sidebar.multiselect(
    "Hisse grubu", list(SEKTORLER), default=list(SEKTORLER)
)
kesif_yil = st.sidebar.select_slider("Keşif penceresi (yıl)", options=[1, 2, 3, 5], value=2)
z_pencere = st.sidebar.slider("Z-skoru penceresi (işlem günü)", 20, 120, 60, step=5)
z_esik = st.sidebar.slider("Z eşiği", 1.0, 3.0, 2.0, step=0.1)
p_esik = st.sidebar.select_slider("p-değeri eşiği", options=[0.01, 0.05, 0.10], value=0.05)
kor_esik = st.sidebar.slider("Korelasyon eşiği", 0.50, 0.95, 0.70, step=0.05)
bh = st.sidebar.checkbox(
    "Benjamini-Hochberg düzeltmesi",
    value=False,
    help="Çok sayıda çift test edince bazıları şans eseri 'eşbütünleşik' çıkar. "
         "BH, p-değerlerini bu çoklu teste göre düzeltir.",
)
st.sidebar.caption("Eğitim amaçlı bir projedir, yatırım tavsiyesi değildir.")

st.title("📈 Pairs Trading Tarayıcı")

if not secili_sektorler:
    st.warning("Soldaki menüden en az bir hisse grubu seç.")
    st.stop()

# ---------------------------------------------------------------------------
# Veri: 30 hissenin hepsi bir kez indirilir, sektör seçimi sadece süzer
# ---------------------------------------------------------------------------
try:
    with st.spinner("Fiyat verisi indiriliyor (ilk açılışta bir-iki dakika sürebilir)..."):
        tum_fiyatlar = veri_yukle(tuple(semboller(list(SEKTORLER))))
except Exception as hata:
    st.error(f"Veri indirilemedi: {hata}")
    st.stop()

eksik = [s for s in semboller(secili_sektorler) if s not in tum_fiyatlar.columns]
if eksik:
    st.warning(f"Şu hisselerin verisi gelmedi, taramaya alınmadı: {', '.join(eksik)}")

ciftler = tuple(sektor_ici_ciftler(secili_sektorler))

# ---------------------------------------------------------------------------
# Tarama: bu ay ve geçen ay (karşılaştırma için)
# ---------------------------------------------------------------------------
with st.spinner("Çiftler taranıyor..."):
    sonuc, huni, kesif_tarihi = tara(tum_fiyatlar, ciftler, kesif_yil, kor_esik, p_esik, 0)
    onceki, _, _ = tara(tum_fiyatlar, ciftler, kesif_yil, kor_esik, p_esik, 1)

p_kolon = "p_bh" if bh else "p"
izleme = sonuc[sonuc[p_kolon] < p_esik].copy()
onceki_izleme = onceki[onceki[p_kolon] < p_esik]

# Her izlenen çiftin güncel z-skoru
guncel_z = []
for _, s in izleme.iterrows():
    _, z = spread_ve_z(tum_fiyatlar, s["y"], s["x"], s["alfa"], s["beta"], z_pencere)
    z = z.dropna()
    guncel_z.append(z.iloc[-1] if len(z) else np.nan)
izleme["z"] = guncel_z
izleme["durum"] = [durum_etiketi(z, z_esik) for z in izleme["z"]]
izleme = izleme.sort_values("z", key=lambda seri: seri.abs(), ascending=False).reset_index(drop=True)

# ---------------------------------------------------------------------------
# Huni sayacı + geçen ayla kıyas
# ---------------------------------------------------------------------------
fark = len(izleme) - len(onceki_izleme)
asamalar = [
    f"{huni['olasi']} olası çift",
    f"{huni['korele']} korele (> {kor_esik:.2f})",
    f"{huni['esbutunlesik']} eşbütünleşik",
]
if bh:
    asamalar.append(f"{huni['bh']} BH sonrası")
st.markdown("### " + " → ".join(asamalar) + f" ({fark:+d})")

kesif_baslangic = kesif_tarihi - pd.DateOffset(years=kesif_yil)
st.caption(
    f"Keşif dönemi: {kesif_baslangic:%d.%m.%Y} – {kesif_tarihi - pd.Timedelta(days=1):%d.%m.%Y} · "
    f"Son güncelleme: {kesif_tarihi:%d.%m.%Y} · "
    f"Son kapanış verisi: {tum_fiyatlar.index[-1]:%d.%m.%Y}"
)

yeni = sorted(set(izleme["cift"]) - set(onceki_izleme["cift"]))
cikan = sorted(set(onceki_izleme["cift"]) - set(izleme["cift"]))
st.caption(
    f"Geçen ay: {len(onceki_izleme)} çift · "
    f"Yeni giren: {', '.join(yeni) if yeni else 'yok'} · "
    f"Çıkan: {', '.join(cikan) if cikan else 'yok'}"
)
if not bh:
    st.caption(f"BH düzeltmesi açılsaydı {huni['bh']} çift kalırdı.")

# ---------------------------------------------------------------------------
# Aktif ve ısınan sinyaller
# ---------------------------------------------------------------------------
aktif = izleme[izleme["z"].abs() > z_esik]
isinan = izleme[(izleme["z"].abs() > 0.75 * z_esik) & (izleme["z"].abs() <= z_esik)]

sol, sag = st.columns(2)
with sol:
    st.subheader(f"🔔 Aktif sinyaller (|z| > {z_esik:.1f}): {len(aktif)}")
    if aktif.empty:
        st.caption("Şu an eşiği aşan çift yok.")
    for _, s in aktif.iterrows():
        st.markdown(f"**{s['cift']}** · z = {s['z']:+.2f} · {yorum(s)}")
with sag:
    st.subheader(f"🌡️ Isınan çiftler ({0.75 * z_esik:.1f} < |z| < {z_esik:.1f}): {len(isinan)}")
    if isinan.empty:
        st.caption("Şu an ısınan çift yok.")
    for _, s in isinan.iterrows():
        st.markdown(f"**{s['cift']}** · z = {s['z']:+.2f} · {yorum(s)}")

# ---------------------------------------------------------------------------
# İzleme listesi
# ---------------------------------------------------------------------------
st.subheader(f"📋 İzleme listesi ({len(izleme)} çift)")
secilen = None
if izleme.empty:
    st.info(
        "Bu ayarlarla eşbütünleşme testini geçen çift yok. "
        "Keşif penceresini uzatmayı ya da p-değeri eşiğini gevşetmeyi deneyebilirsin."
    )
else:
    gosterim = pd.DataFrame({
        "Çift": izleme["cift"],
        "Durum": izleme["durum"],
        "Güncel z": izleme["z"].round(2),
        "Korelasyon": izleme["korelasyon"].round(3),
        "EG p-değeri": izleme["p"].round(4),
        "BH p-değeri": izleme["p_bh"].round(4),
        "Hedge oranı (β)": izleme["beta"].round(3),
        "Half-life (gün)": izleme["half_life"].round(1),
    })
    olay = st.dataframe(gosterim, hide_index=True, on_select="rerun", selection_mode="single-row")
    st.caption("Detayını görmek için tablodan bir satırın solundaki kutuya tıkla.")
    secili_satirlar = olay.selection.rows
    if secili_satirlar:
        secilen = izleme.iloc[secili_satirlar[0]]

# ---------------------------------------------------------------------------
# Korelasyon ısı haritası + (ileride) aylık geçmiş
# ---------------------------------------------------------------------------
b1, b2 = st.columns(2)
with b1:
    korelasyon_ac = st.toggle("Korelasyonları göster")
with b2:
    st.button("Aylık geçmiş", disabled=True, help="v1.1'de geliyor")

if korelasyon_ac:
    mevcut = [s for s in semboller(secili_sektorler) if s in tum_fiyatlar.columns]
    donem, _ = kesif_donemi(tum_fiyatlar[mevcut], kesif_yil)
    kor = korelasyon_matrisi(donem)
    fig_kor = px.imshow(
        kor, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
        aspect="auto", title="Günlük getiri korelasyonları (keşif dönemi)",
    )
    fig_kor.update_layout(height=max(500, 28 * len(kor)))
    st.plotly_chart(fig_kor)

# ---------------------------------------------------------------------------
# Çift detayı
# ---------------------------------------------------------------------------
if secilen is not None:
    s = secilen
    y, x = s["y"], s["x"]
    st.divider()
    st.header(f"🔍 {sirket_adi(y)} ({y}) / {sirket_adi(x)} ({x})")

    m = st.columns(5)
    m[0].metric("Korelasyon", f"{s['korelasyon']:.3f}")
    m[1].metric("EG p-değeri", f"{s['p']:.4f}")
    m[2].metric("Hedge oranı (β)", f"{s['beta']:.3f}")
    m[3].metric("Half-life", "∞" if np.isinf(s["half_life"]) else f"{s['half_life']:.1f} gün")
    m[4].metric("Güncel z", f"{s['z']:+.2f}")
    if pd.notna(s["z"]) and abs(s["z"]) > 0.75 * z_esik:
        st.info(yorum(s))

    # Normalize fiyatlar (keşif başı = 1)
    ikili = tum_fiyatlar.loc[tum_fiyatlar.index >= kesif_baslangic, [y, x]].dropna()
    normal = ikili / ikili.iloc[0]
    fig1 = px.line(normal, title="Normalize fiyatlar (keşif başı = 1)")
    fig1.update_layout(xaxis_title="Tarih", yaxis_title="Normalize fiyat", legend_title="Hisse")
    kesif_cizgisi(fig1, kesif_tarihi)
    st.plotly_chart(fig1)

    # Z-skoru ve geçmiş sinyaller
    _, z = spread_ve_z(tum_fiyatlar, y, x, s["alfa"], s["beta"], z_pencere)
    z = z[z.index >= kesif_baslangic]
    sinyaller = sinyal_noktalari(z, z_esik)
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=z.index, y=z.values, mode="lines", name="z-skoru"))
    fig2.add_trace(go.Scatter(
        x=sinyaller.index, y=sinyaller.values, mode="markers", name="Sinyal",
        marker=dict(size=9, color="red"),
    ))
    for seviye in (z_esik, -z_esik):
        fig2.add_hline(y=seviye, line_dash="dash", line_color="red")
    fig2.add_hline(y=0, line_color="gray")
    fig2.update_layout(
        title=f"Z-skoru ({z_pencere} günlük pencere, ±{z_esik:.1f} bantları)",
        xaxis_title="Tarih", yaxis_title="z",
    )
    kesif_cizgisi(fig2, kesif_tarihi)
    st.plotly_chart(fig2)
    st.caption(
        f"Spread = log({y}) − {s['beta']:.3f}·log({x}) − {s['alfa']:.3f}. "
        "β keşif dönemindeki veriyle tahmin edildi; 'keşif sonu' çizgisinin sağı örneklem dışıdır. "
        f"Kırmızı noktalar: |z|'nin {z_esik:.1f} eşiğini dışarı doğru geçtiği günler."
    )
