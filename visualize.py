from collections import defaultdict

import cv2
import numpy as np
import pandas as pd

# ---------------- GÖRSEL AYARLAR (1080p'ye göre) ----------------
CAR_THICKNESS = 4        # araç köşe çizgisi kalınlığı
CAR_LINE_RATIO = 0.25    # köşe çizgi uzunluğu = kutu boyutunun %25'i
PLATE_THICKNESS = 2      # kırmızı plaka kutusu kalınlığı
CROP_H = 100             # büyütülmüş plaka kırpmasının yüksekliği
BAND_H = 60              # metnin yazıldığı beyaz bandın yüksekliği
GAP = 15                 # panel ile araç kutusu arası boşluk
FONT_SCALE = 1.5         # metin boyutu
FONT_THICK = 4           # metin kalınlığı
# -----------------------------------------------------------------

# ---------------- OYLAMA AYARLARI ----------------
MIN_SKOR = 0.3            # bundan düşük okumalar oylamaya girmez
USS = 3                   # skorun üssü: büyüdükçe güven, sıklığa göre daha baskın olur
MIN_OKUMA = 2             # bir araç için en az kaç okuma gerekir (tek seferlik şüpheli okumaları eler)
ONCELIKLI_IL = {'34'}     # bu il kodlarıyla başlayan okumalar tam ağırlık alır
DIGER_IL_CARPANI = 0.5    # diğer il kodlarının ağırlık çarpanı (yasaklamaz, sadece zayıflatır)
# --------------------------------------------------

# ---------------- DOĞRULUK ÖLÇÜMÜ (isteğe bağlı) ----------------
# İlk çalıştırmada terminalde araç ID'lerini gör, sonra gerçek plakaları buraya yaz:
# GERCEK = {1: "34GT2850", 2: "34GJK657"}
GERCEK = {}
# ----------------------------------------------------------------


def draw_border(img, top_left, bottom_right, color=(0, 255, 0), thickness=4, ratio=0.25):
    x1, y1 = top_left
    x2, y2 = bottom_right
    lx = int((x2 - x1) * ratio)
    ly = int((y2 - y1) * ratio)

    cv2.line(img, (x1, y1), (x1, y1 + ly), color, thickness)
    cv2.line(img, (x1, y1), (x1 + lx, y1), color, thickness)

    cv2.line(img, (x1, y2), (x1, y2 - ly), color, thickness)
    cv2.line(img, (x1, y2), (x1 + lx, y2), color, thickness)

    cv2.line(img, (x2, y1), (x2 - lx, y1), color, thickness)
    cv2.line(img, (x2, y1), (x2, y1 + ly), color, thickness)

    cv2.line(img, (x2, y2), (x2, y2 - ly), color, thickness)
    cv2.line(img, (x2, y2), (x2 - lx, y2), color, thickness)
    return img


def parse_bbox(s):
    """'x1 y1 x2 y2' (eski '[x1 y1 x2 y2]' biçimi de olur) -> 4 float."""
    return tuple(map(float, s.replace('[', '').replace(']', '').replace(',', ' ').split()))


def make_panel(crop, text):
    """Beyaz bant (metin) + altında plaka kırpması içeren tek bir görsel üretir."""
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, FONT_THICK)
    W = max(crop.shape[1], tw + 30)
    panel = np.full((BAND_H + CROP_H, W, 3), 255, dtype=np.uint8)

    cv2.putText(panel, text, ((W - tw) // 2, (BAND_H + th) // 2),
                cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, (0, 0, 0), FONT_THICK)

    x0 = (W - crop.shape[1]) // 2
    panel[BAND_H:BAND_H + CROP_H, x0:x0 + crop.shape[1]] = crop
    return panel


def paste_clamped(frame, panel, cx, y_bottom):
    """Paneli ekran dışına taşmayacak şekilde yapıştırır."""
    H, W, _ = panel.shape
    fh, fw, _ = frame.shape
    x0 = int(cx - W / 2)
    y0 = int(y_bottom - H)
    x0 = max(0, min(x0, fw - W))
    y0 = max(0, min(y0, fh - H))
    frame[y0:y0 + H, x0:x0 + W] = panel


def sekil_of(text):
    """'34THP99' -> 'RRHHHRR' (harf/rakam yapısı). Aynı yapıdaki okumalar pozisyon pozisyon oylanır."""
    return ''.join('H' if c.isalpha() else 'R' for c in text)


def karakter_oylamasi(best):
    """Karakter bazlı ağırlıklı oylama.
    1) Okumalar harf/rakam yapısına göre gruplanır, en ağırlıklı yapı seçilir.
    2) O gruptaki her pozisyon kendi içinde skor**USS ağırlığıyla oylanır.
    Döndürür: (plaka, kazanan grup okumaları, grup ağırlıkları, ortalama karakter güveni)"""
    okumalar = []
    for _, r in best.iterrows():
        text = r['license_number']
        agirlik = r['license_number_score'] ** USS
        if text[:2] not in ONCELIKLI_IL:
            agirlik *= DIGER_IL_CARPANI
        okumalar.append({'text': text, 'agirlik': agirlik, 'sekil': sekil_of(text), 'row': r})

    grup_agirlik = defaultdict(float)
    for o in okumalar:
        grup_agirlik[o['sekil']] += o['agirlik']
    sekil = max(grup_agirlik, key=grup_agirlik.get)
    grup = [o for o in okumalar if o['sekil'] == sekil]

    # il kodu (ilk 2 karakter) tek parça olarak oylanır: "34" ile "04" yarışır, "3" ile "0" ayrı ayrı değil
    il_oy = defaultdict(float)
    for o in grup:
        il_oy[o['text'][:2]] += o['agirlik']
    il = max(il_oy, key=il_oy.get)
    il_pay = il_oy[il] / sum(il_oy.values())

    sonuc, paylar = il, [il_pay, il_pay]
    for i in range(2, len(sekil)):
        oy = defaultdict(float)
        for o in grup:
            oy[o['text'][i]] += o['agirlik']
        kazanan = max(oy, key=oy.get)
        sonuc += kazanan
        paylar.append(oy[kazanan] / sum(oy.values()))

    return sonuc, grup, grup_agirlik, float(np.mean(paylar))


results = pd.read_csv('./test_interpolated.csv')

# interpolasyon satırlarında '0' olabiliyor
results['license_number'] = results['license_number'].astype(str)
results['license_number_score'] = pd.to_numeric(results['license_number_score'], errors='coerce').fillna(0)

# videoyu yükle
video_path = 'video_h264.mp4'
cap = cv2.VideoCapture(video_path)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter('./out.mp4', fourcc, fps, (width, height))

# ---------------- ID BİRLEŞTİRME (plaka benzerliğine göre) ----------------
MAX_FARK = 2              # iki ID'nin oylanmış plakaları en fazla kaç karakter farklıysa aynı araç sayılır
MAX_ORTUSME = 10          # iki ID aynı anda en fazla kaç kare görünebilir (fazlaysa farklı araçtır)
MAX_ZAMAN_BOSLUGU = 200   # iki ID arasında en fazla kaç kare boşluk olabilir
# ---------------------------------------------------------------------------


def edit_mesafe(a, b):
    """Levenshtein: iki metin arasındaki en az karakter değişikliği."""
    if len(a) < len(b):
        a, b = b, a
    onceki = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        simdiki = [i]
        for j, cb in enumerate(b, 1):
            simdiki.append(min(onceki[j] + 1, simdiki[j - 1] + 1, onceki[j - 1] + (ca != cb)))
        onceki = simdiki
    return onceki[-1]


# her ID için kare aralığı ve filtrelenmiş okumalar
filtreli, araliklar = {}, {}
for car_id in np.unique(results['car_id']):
    tum = results[results['car_id'] == car_id]
    araliklar[car_id] = (int(tum['frame_nmr'].min()), int(tum['frame_nmr'].max()))
    best = tum[(tum['license_number'] != '0') & (tum['license_number_score'] >= MIN_SKOR)]
    if not best.empty:
        filtreli[car_id] = best

# 1) her ID için geçici plaka
gecici = {cid: karakter_oylamasi(b)[0] for cid, b in filtreli.items()}

# 2) zamanda çakışmayan + plakası benzeyen ID'leri birleştir
kok = {cid: cid for cid in filtreli}


def bul(c):
    while kok[c] != c:
        c = kok[c]
    return c


ids = sorted(filtreli, key=lambda c: araliklar[c][0])
for i, a in enumerate(ids):
    for b in ids[i + 1:]:
        a0, a1 = araliklar[a]
        b0, b1 = araliklar[b]
        ortusme = min(a1, b1) - max(a0, b0)
        bosluk = max(a0, b0) - min(a1, b1)
        if ortusme > MAX_ORTUSME or bosluk > MAX_ZAMAN_BOSLUGU:
            continue
        fark = edit_mesafe(gecici[a], gecici[b])
        if fark <= MAX_FARK and bul(a) != bul(b):
            kok[bul(b)] = bul(a)
            print(f"ID {b} ({gecici[b]}) ile ID {a} ({gecici[a]}) aynı araç sayıldı, fark={fark}")

kumeler = defaultdict(list)
for cid in filtreli:
    kumeler[bul(cid)].append(cid)

# 3) her küme için birleşik okumalarla yeniden oyla
license_plate = {}
secilenler = {}
for kok_id, uyeler in kumeler.items():
    best = pd.concat([filtreli[c] for c in uyeler])
    en_iyi_metin, grup, grup_agirlik, guven = karakter_oylamasi(best)

    aralik_metni = ", ".join(f"{c}: kare {araliklar[c][0]}-{araliklar[c][1]}" for c in uyeler)
    print(f"araç(lar) [{aralik_metni}]: seçilen = {en_iyi_metin} | {len(grup)} okuma, "
          f"karakter uyumu %{guven * 100:.0f} | "
          + ", ".join(f"{m}: {n}" for m, n in best['license_number'].value_counts().items()))

    if len(grup) < MIN_OKUMA:
        print(f"   -> atlandı: yetersiz okuma ({len(grup)} < {MIN_OKUMA})")
        continue

    # temsilci kare: kazanan plakaya en çok benzeyen, eşitlikte en yüksek skorlu okuma
    def benzerlik(o):
        uyusma = sum(a == b for a, b in zip(o['text'], en_iyi_metin))
        return (uyusma, o['row']['license_number_score'])
    row = max(grup, key=benzerlik)['row']

    cap.set(cv2.CAP_PROP_POS_FRAMES, int(row['frame_nmr']))
    ret, frame = cap.read()
    if not ret:
        continue

    x1, y1, x2, y2 = parse_bbox(row['license_plate_bbox'])
    crop = frame[int(y1):int(y2), int(x1):int(x2), :]
    if crop.size == 0:
        continue
    crop = cv2.resize(crop, (int((x2 - x1) * CROP_H / (y2 - y1)), CROP_H))

    panel = make_panel(crop, en_iyi_metin)
    for c in uyeler:                      # kümedeki tüm ID'ler aynı paneli gösterir
        license_plate[c] = {'panel': panel}
        secilenler[int(c)] = en_iyi_metin

# ---- doğruluk ölçümü ----
if GERCEK:
    dogru = sum(secilenler.get(k) == v for k, v in GERCEK.items())
    print(f"\nDOĞRULUK: {dogru}/{len(GERCEK)} (%{100 * dogru / len(GERCEK):.0f})")
    for k, v in GERCEK.items():
        if secilenler.get(k) != v:
            print(f"   araç {k}: gerçek = {v}, bulunan = {secilenler.get(k)}")

# ---- videoyu baştan oku ve çiz ----
frame_nmr = -1
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

# karelere göre grupla (her karede tüm CSV'yi taramamak için)
kare_gruplari = {k: g for k, g in results.groupby('frame_nmr')}

ret = True
while ret:
    ret, frame = cap.read()
    frame_nmr += 1
    if ret:
        df_ = kare_gruplari.get(frame_nmr)
        if df_ is not None:
            for row_indx in range(len(df_)):
                car_id = df_.iloc[row_indx]['car_id']
                if car_id not in license_plate:
                    continue

                # araç
                car_x1, car_y1, car_x2, car_y2 = parse_bbox(df_.iloc[row_indx]['car_bbox'])
                draw_border(frame, (int(car_x1), int(car_y1)), (int(car_x2), int(car_y2)),
                            (0, 255, 0), CAR_THICKNESS, CAR_LINE_RATIO)

                # plaka kutusu
                x1, y1, x2, y2 = parse_bbox(df_.iloc[row_indx]['license_plate_bbox'])
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), PLATE_THICKNESS)

                # plaka paneli: aracın üstünde, ortalı
                paste_clamped(frame, license_plate[car_id]['panel'],
                              (car_x1 + car_x2) / 2, car_y1 - GAP)

        out.write(frame)

out.release()
cap.release()
print("out.mp4 yazıldı")