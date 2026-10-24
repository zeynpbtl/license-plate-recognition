from ultralytics import YOLO
import cv2
import numpy as np
from sort.sort import *
from util import get_car, read_license_plate, write_csv

# ---------------- AYARLAR ----------------
VIDEO_YOLU = "video_h264.mp4"   # işlenecek video (visualize.py ile aynı olmalı)
MAX_FRAME = None                # None = tüm video; test için örn. 100 yaz
OLCEK = 3                       # plaka kırpmasını kaç kat büyütelim (OCR için)
ESIKLEME = "inv64"              # "inv64": senin çalışan yöntemin
                                # "otsu"  : otomatik eşik
                                # "yok"   : ham gri görüntü
VEHICLES = [2, 3, 5, 7]         # COCO: car, motorcycle, bus, truck
# ------------------------------------------

results = {}

# araç takipçisi (aynı araca kareler boyunca aynı ID verir)
mot_tracker = Sort()

# modelleri yükle
coco_model = YOLO('yolo26n.pt')              # araç tespiti
license_plate_detector = YOLO('best.pt')     # plaka tespiti (kendi eğittiğimiz)

# videoyu yükle
cap = cv2.VideoCapture(VIDEO_YOLU)
print("video açıldı mı:", cap.isOpened())


def plaka_isle(crop):
    """Plaka kırpmasını OCR'a hazırlar: büyüt, griye çevir, eşikle."""
    crop = cv2.resize(crop, None, fx=OLCEK, fy=OLCEK, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

    if ESIKLEME == "inv64":
        _, out = cv2.threshold(gray, 64, 255, cv2.THRESH_BINARY_INV)
    elif ESIKLEME == "otsu":
        _, out = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        out = gray
    return out


# kareleri oku
frame_nmr = -1
ret = True
while ret:
    frame_nmr += 1
    ret, frame = cap.read()

    # video bitti ya da test sınırına ulaşıldı
    if not ret or (MAX_FRAME is not None and frame_nmr >= MAX_FRAME):
        break

    results[frame_nmr] = {}

    # ---- araçları tespit et ----
    detections = coco_model(frame, verbose=False)[0]
    detections_ = []
    for detection in detections.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = detection
        if int(class_id) in VEHICLES:               # sadece araç sınıfları
            detections_.append([x1, y1, x2, y2, score])

    # ---- araçları takip et ----
    # araç yoksa boş (0,5) dizi ver, yoksa SORT hata verebilir
    dets = np.asarray(detections_) if detections_ else np.empty((0, 5))
    track_ids = mot_tracker.update(dets)

    # ---- plakaları tespit et ----
    license_plates = license_plate_detector(frame, verbose=False)[0]
    for license_plate in license_plates.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = license_plate

        # plakayı hangi araca ait olduğuna göre eşleştir
        xcar1, ycar1, xcar2, ycar2, car_id = get_car(license_plate, track_ids)

        if car_id != -1:                            # eşleşen araç bulunduysa
            # plakayı kırp
            license_plate_crop = frame[int(y1):int(y2), int(x1):int(x2), :]
            if license_plate_crop.size == 0:        # boş kırpmayı atla
                continue

            # OCR için işle (büyüt + gri + eşik)
            license_plate_crop_thresh = plaka_isle(license_plate_crop)

            # plaka metnini oku
            license_plate_text, license_plate_text_score = read_license_plate(
                license_plate_crop_thresh)

            # sadece formata uyan okumaları kaydet
            if license_plate_text is not None:
                results[frame_nmr][car_id] = {
                    'car': {'bbox': [xcar1, ycar1, xcar2, ycar2]},
                    'license_plate': {'bbox': [x1, y1, x2, y2],  # plaka kutusu (orijinal kare koordinatı)
                                      'text': license_plate_text,
                                      'bbox_score': score,                      # tespit güveni
                                      'text_score': license_plate_text_score}}  # OCR güveni

    # ilerleme göstergesi
    if frame_nmr % 20 == 0:
        print(f"kare {frame_nmr} işlendi")

cap.release()

# sonuçları CSV'ye yaz
write_csv(results, './test.csv')
print("test.csv yazıldı")