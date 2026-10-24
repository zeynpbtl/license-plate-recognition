import csv
from collections import defaultdict

import numpy as np


def parse_bbox(s):
    """'x1 y1 x2 y2' (eski '[x1 y1 x2 y2]' biçimi de olur) -> 4 float."""
    return list(map(float, s.replace('[', '').replace(']', '').replace(',', ' ').split()))


def fmt(b):
    return ' '.join(f'{v:.2f}' for v in b)


def interpolate_bounding_boxes(data):
    # araç -> {kare: satır} sözlüğü (satır taramak yerine doğrudan erişim)
    por_arac = defaultdict(dict)
    for row in data:
        por_arac[int(float(row['car_id']))][int(row['frame_nmr'])] = row

    interpolated_data = []
    for car_id in sorted(por_arac):
        kareler = por_arac[car_id]
        f_list = sorted(kareler)
        first, last = f_list[0], f_list[-1]

        car_pts = np.array([parse_bbox(kareler[f]['car_bbox']) for f in f_list])
        plate_pts = np.array([parse_bbox(kareler[f]['license_plate_bbox']) for f in f_list])

        tum_kareler = np.arange(first, last + 1)
        # eksik karelerin kutularını doğrusal olarak doldur
        car_i = np.column_stack([np.interp(tum_kareler, f_list, car_pts[:, k]) for k in range(4)])
        plate_i = np.column_stack([np.interp(tum_kareler, f_list, plate_pts[:, k]) for k in range(4)])

        for idx, frame_number in enumerate(tum_kareler):
            frame_number = int(frame_number)
            row = {'frame_nmr': frame_number,
                   'car_id': car_id,
                   'car_bbox': fmt(car_i[idx]),
                   'license_plate_bbox': fmt(plate_i[idx])}

            orijinal = kareler.get(frame_number)
            if orijinal is None:
                # interpolasyonla eklenen satır
                row['license_plate_bbox_score'] = '0'
                row['license_number'] = '0'
                row['license_number_score'] = '0'
            else:
                row['license_plate_bbox_score'] = orijinal['license_plate_bbox_score']
                row['license_number'] = orijinal['license_number']
                row['license_number_score'] = orijinal['license_number_score']

            interpolated_data.append(row)

    return interpolated_data


# ---------------- ID BİRLEŞTİRME AYARLARI ----------------
MAX_BOSLUK = 30   # bir ID bittikten en fazla kaç kare sonra başlayan yeni ID devam sayılır
MIN_IOU = 0.3     # eski ID'nin son kutusu ile yeni ID'nin ilk kutusunun en az örtüşmesi
# ---------------------------------------------------------


def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def idleri_birlestir(data):
    """Takipçinin aynı araca verdiği ardışık ID'leri tek ID'ye indirir."""
    bilgi = {}
    for row in data:
        cid = int(float(row['car_id']))
        f = int(row['frame_nmr'])
        box = parse_bbox(row['car_bbox'])
        b = bilgi.setdefault(cid, {'ilk': f, 'son': f, 'ilk_box': box, 'son_box': box})
        if f < b['ilk']:
            b['ilk'], b['ilk_box'] = f, box
        if f > b['son']:
            b['son'], b['son_box'] = f, box

    kok = {cid: cid for cid in bilgi}

    def bul(c):
        while kok[c] != c:
            c = kok[c]
        return c

    devam_etti = set()   # her ID'nin en fazla bir halefi olur
    for cid in sorted(bilgi, key=lambda c: bilgi[c]['ilk']):
        b = bilgi[cid]
        en_iyi, en_iou = None, MIN_IOU
        for aday, a in bilgi.items():
            if aday == cid or aday in devam_etti:
                continue
            bosluk = b['ilk'] - a['son']
            if not 0 < bosluk <= MAX_BOSLUK:
                continue
            v = iou(a['son_box'], b['ilk_box'])
            if v >= en_iou:
                en_iyi, en_iou = aday, v
        if en_iyi is not None:
            kok[cid] = bul(en_iyi)
            devam_etti.add(en_iyi)
            print(f"ID {cid} -> ID {bul(en_iyi)} ile birleştirildi (IoU {en_iou:.2f})")

    # yeniden etiketle; aynı karede çakışma olursa okuma skoru yüksek olan kalsın
    secilen = {}
    for row in data:
        row['car_id'] = str(bul(int(float(row['car_id']))))
        anahtar = (row['car_id'], int(row['frame_nmr']))
        eski = secilen.get(anahtar)
        if eski is None or float(row['license_number_score']) > float(eski['license_number_score']):
            secilen[anahtar] = row
    return list(secilen.values())


# CSV'yi oku
with open('test.csv', 'r') as file:
    data = list(csv.DictReader(file))

data = idleri_birlestir(data)

interpolated_data = interpolate_bounding_boxes(data)

header = ['frame_nmr', 'car_id', 'car_bbox', 'license_plate_bbox',
          'license_plate_bbox_score', 'license_number', 'license_number_score']
with open('test_interpolated.csv', 'w', newline='') as file:
    writer = csv.DictWriter(file, fieldnames=header)
    writer.writeheader()
    writer.writerows(interpolated_data)

print("test_interpolated.csv yazıldı")