import ast

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


def draw_border(img, top_left, bottom_right, color=(0, 255, 0), thickness=4, ratio=0.25):
    x1, y1 = top_left
    x2, y2 = bottom_right
    # çizgi uzunluğu kutunun boyutuna göre (küçük araçta kutuyu kapatmasın)
    lx = int((x2 - x1) * ratio)
    ly = int((y2 - y1) * ratio)

    cv2.line(img, (x1, y1), (x1, y1 + ly), color, thickness)  # sol-üst
    cv2.line(img, (x1, y1), (x1 + lx, y1), color, thickness)

    cv2.line(img, (x1, y2), (x1, y2 - ly), color, thickness)  # sol-alt
    cv2.line(img, (x1, y2), (x1 + lx, y2), color, thickness)

    cv2.line(img, (x2, y1), (x2 - lx, y1), color, thickness)  # sağ-üst
    cv2.line(img, (x2, y1), (x2, y1 + ly), color, thickness)

    cv2.line(img, (x2, y2), (x2, y2 - ly), color, thickness)  # sağ-alt
    cv2.line(img, (x2, y2), (x2 - lx, y2), color, thickness)
    return img


def parse_bbox(s):
    """CSV'deki '[ 12.3  45.6 ...]' metnini 4 sayıya çevirir."""
    s = s.replace('[ ', '[').replace('   ', ' ').replace('  ', ' ').replace(' ', ',')
    return ast.literal_eval(s)


def make_panel(crop, text):
    """Beyaz bant (metin) + altında plaka kırpması içeren tek bir görsel üretir."""
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, FONT_THICK)
    W = max(crop.shape[1], tw + 30)               # panel, metin ve kırpmadan geniş olanı kadar
    panel = np.full((BAND_H + CROP_H, W, 3), 255, dtype=np.uint8)

    # metin: beyaz bandın ortasına
    cv2.putText(panel, text, ((W - tw) // 2, (BAND_H + th) // 2),
                cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, (0, 0, 0), FONT_THICK)

    # kırpma: bandın altına, yatayda ortalı
    x0 = (W - crop.shape[1]) // 2
    panel[BAND_H:BAND_H + CROP_H, x0:x0 + crop.shape[1]] = crop
    return panel


def paste_clamped(frame, panel, cx, y_bottom):
    """Paneli ekran dışına taşmayacak şekilde yapıştırır."""
    H, W, _ = panel.shape
    fh, fw, _ = frame.shape
    x0 = int(cx - W / 2)
    y0 = int(y_bottom - H)
    x0 = max(0, min(x0, fw - W))      # yatayda ekranın içinde tut
    y0 = max(0, min(y0, fh - H))      # dikeyde ekranın içinde tut
    frame[y0:y0 + H, x0:x0 + W] = panel


results = pd.read_csv('./test_interpolated.csv')

# sayısal/metin sütunlarını güvene al (interpolasyon satırlarında '0' olabiliyor)
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

# ---- OYLAMA AYARLARI ----
MIN_SKOR = 0.3     # bundan düşük okumalar oylamaya girmez
USS = 3            # skorun üssü: büyüdükçe güven, sıklığa göre daha baskın olur
# -------------------------

license_plate = {}
for car_id in np.unique(results['car_id']):
    best = results[results['car_id'] == car_id]

    # interpolasyon satırlarını ve düşük güvenli okumaları ele
    best = best[(best['license_number'] != '0') & (best['license_number_score'] >= MIN_SKOR)]
    if best.empty:
        continue

    # her metin için skor**USS toplamı
    agirlik = best['license_number_score'] ** USS
    toplam = agirlik.groupby(best['license_number']).sum()
    en_iyi_metin = toplam.idxmax()

    adet = best['license_number'].value_counts()
    print(f"araç {car_id}: seçilen = {en_iyi_metin} | "
          + ", ".join(f"{m}: {adet[m]} okuma, ağırlık {w:.2f}"
                      for m, w in toplam.sort_values(ascending=False).items()))

    # (geri kalanı aynı: kazanan metnin en yüksek skorlu karesi, kırpma, panel)
    adaylar = best[best['license_number'] == en_iyi_metin]
    row = adaylar.loc[adaylar['license_number_score'].idxmax()]

    cap.set(cv2.CAP_PROP_POS_FRAMES, row['frame_nmr'])
    ret, frame = cap.read()
    if not ret:
        continue

    x1, y1, x2, y2 = parse_bbox(row['license_plate_bbox'])
    crop = frame[int(y1):int(y2), int(x1):int(x2), :]
    if crop.size == 0:
        continue
    crop = cv2.resize(crop, (int((x2 - x1) * CROP_H / (y2 - y1)), CROP_H))

    license_plate[car_id] = {'panel': make_panel(crop, en_iyi_metin)}

# ---- videoyu baştan oku ve çiz ----
frame_nmr = -1
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

ret = True
while ret:
    ret, frame = cap.read()
    frame_nmr += 1
    if ret:
        df_ = results[results['frame_nmr'] == frame_nmr]
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