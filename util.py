import csv
import re

import easyocr

# Initialize the OCR reader
reader = easyocr.Reader(['en'], gpu=False)

# ---------------- AYARLAR ----------------
# OCR'ın okuyabileceği karakterler (O, Q, W, X yok; I bilerek dahil).
ALLOWLIST = 'ABCDEFGHIJKLMNPRSTUVYZ0123456789'

MIN_OCR_SKOR = 0.25   # bundan düşük güvenli okumalar baştan çöpe atılır
MAX_DEGISIM = 1       # format düzeltmesinde en fazla kaç karakter değiştirilebilir

# Harf sayısına göre izin verilen rakam sayısı (min, max).
# NOT: Bu aralıklar benim bilgime göre; yanlış bulursan sadece burayı düzelt.
RAKAM_ARALIK = {1: (3, 4), 2: (3, 4), 3: (2, 3)}
# ------------------------------------------

dict_char_to_int = {'O': '0', 'Q': '0', 'D': '0', 'I': '1', 'L': '1',
                    'Z': '2', 'S': '5', 'B': '8', 'G': '6', 'A': '4'}
dict_int_to_char = {'0': 'D', '1': 'I', '2': 'Z', '5': 'S',
                    '8': 'B', '6': 'G', '4': 'A'}

HARF = 'ABCDEFGHIJKLMNPRSTUVYZ'
PATTERN = re.compile(rf'(0[1-9]|[1-7][0-9]|8[01])([{HARF}]{{1,3}})(\d{{2,4}})')


def write_csv(results, output_path):
    """Sonuçları CSV'ye yazar. Kutular 'x1 y1 x2 y2' (boşlukla ayrılmış) biçiminde."""
    def bb(b):
        return ' '.join(f'{float(v):.2f}' for v in b)

    with open(output_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['frame_nmr', 'car_id', 'car_bbox', 'license_plate_bbox',
                    'license_plate_bbox_score', 'license_number', 'license_number_score'])
        for frame_nmr, cars in results.items():
            for car_id, r in cars.items():
                if 'car' in r and 'license_plate' in r and 'text' in r['license_plate']:
                    lp = r['license_plate']
                    w.writerow([frame_nmr, int(car_id), bb(r['car']['bbox']), bb(lp['bbox']),
                                lp['bbox_score'], lp['text'], lp['text_score']])


def gecerli_plaka(text):
    """Tam format kontrolü: il kodu + 1-3 harf + harf sayısına uygun rakam sayısı."""
    m = PATTERN.fullmatch(text)
    if not m:
        return False
    lo, hi = RAKAM_ARALIK[len(m.group(2))]
    return lo <= len(m.group(3)) <= hi


def license_complies_format(text):
    return format_license(text) is not None


def format_license(text):
    """Pozisyona göre düzeltir (en fazla MAX_DEGISIM karakter değişir).
    Uygun biçim bulamazsa None döner."""
    if gecerli_plaka(text):
        return text

    en_iyi, en_az = None, 99
    for k in (1, 2, 3):                      # harf sayısı
        lo, hi = RAKAM_ARALIK[k]
        if not lo <= len(text) - 2 - k <= hi:
            continue
        yeni, degisim = [], 0
        for i, c in enumerate(text):
            if 2 <= i < 2 + k:               # harf olmalı
                n = dict_int_to_char.get(c, c)
            else:                            # rakam olmalı
                n = dict_char_to_int.get(c, c)
            degisim += (n != c)
            yeni.append(n)
        if degisim > MAX_DEGISIM:
            continue
        aday = ''.join(yeni)
        if gecerli_plaka(aday) and degisim < en_az:
            en_iyi, en_az = aday, degisim
    return en_iyi


def read_license_plate(license_plate_crop):
    """Kırpılmış plaka görüntüsünden (biçimlendirilmiş metin, skor) döndürür."""
    detections = reader.readtext(license_plate_crop, allowlist=ALLOWLIST)
    if not detections:
        return None, None

    # parçaları soldan sağa sırala
    detections = sorted(detections, key=lambda d: d[0][0][0])

    # adaylar: önce her parça tek başına, sonra hepsinin birleşimi
    adaylar = [(d[1], d[2]) for d in detections]
    if len(detections) > 1:
        birlesik = ''.join(d[1] for d in detections)
        ortalama = sum(d[2] for d in detections) / len(detections)
        adaylar.append((birlesik, ortalama))

    for text, score in adaylar:
        if score < MIN_OCR_SKOR:
            continue
        text = text.upper().replace(' ', '')
        text = re.sub(r'[^A-Z0-9]', '', text)
        if text.startswith('TR'):            # mavi şerit
            text = text[2:]

        duzgun = format_license(text)
        if duzgun is not None:
            return duzgun, score

    return None, None


def get_car(license_plate, vehicle_track_ids):
    """Plakayı içeren araçlar arasından en küçük alanlıyı seçer (iç içe kutularda doğru eşleşme)."""
    x1, y1, x2, y2, score, class_id = license_plate

    en_iyi, en_kucuk_alan = None, None
    for j in range(len(vehicle_track_ids)):
        xcar1, ycar1, xcar2, ycar2, car_id = vehicle_track_ids[j]
        if x1 > xcar1 and y1 > ycar1 and x2 < xcar2 and y2 < ycar2:
            alan = (xcar2 - xcar1) * (ycar2 - ycar1)
            if en_kucuk_alan is None or alan < en_kucuk_alan:
                en_iyi, en_kucuk_alan = vehicle_track_ids[j], alan

    if en_iyi is not None:
        return en_iyi
    return -1, -1, -1, -1, -1