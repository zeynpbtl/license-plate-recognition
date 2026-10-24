from ultralytics import YOLO
import cv2
import numpy as np
from util import get_car, read_license_plate, write_csv

# ---------------- AYARLAR ----------------
VIDEO_YOLU = "video_h264.mp4"   # işlenecek video (visualize.py ile aynı olmalı)
MAX_FRAME = None                # None = tüm video; test için örn. 100 yaz
OLCEK = 3                       # plaka kırpmasını kaç kat büyütelim (OCR için)
ESIKLEME = "coklu"              # "coklu"  : inv64 + clahe + otsu denenir, en güvenli okuma seçilir (yavaş)
                                # "inv64" | "otsu" | "clahe" | "adaptif" | "yok" : tek yöntem
VEHICLES = [2, 3, 5, 7]         # COCO: car, motorcycle, bus, truck
TAKIPCI = "sort"           # "bytetrack" (ultralytics) ya da "sort"
MIN_PLAKA_YUK = 18              # orijinal karede plaka yüksekliği (px) bundan küçükse OCR yapma
MIN_PLAKA_GEN = 45              # aynısı genişlik için
# ------------------------------------------

results = {}

# modelleri yükle
coco_model = YOLO('yolo26n.pt')              # araç tespiti
license_plate_detector = YOLO('best.pt')     # plaka tespiti (kendi eğittiğimiz)

if TAKIPCI == "sort":
    from sort.sort import Sort
    mot_tracker = Sort()

# videoyu yükle
cap = cv2.VideoCapture(VIDEO_YOLU)
print("video açıldı mı:", cap.isOpened())

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))


def tek_yontem(gray, ad):
    if ad == "inv64":
        return cv2.threshold(gray, 64, 255, cv2.THRESH_BINARY_INV)[1]
    if ad == "otsu":
        blur = cv2.GaussianBlur(gray, (3, 3), 0)
        return cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    if ad == "clahe":
        return clahe.apply(gray)
    if ad == "adaptif":
        blur = cv2.GaussianBlur(gray, (3, 3), 0)
        return cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY, 31, 10)
    return gray


def plaka_varyantlari(crop):
    """Plaka kırpmasını OCR'a hazırlar: büyüt, griye çevir, seçilen yöntem(ler)le işle."""
    crop = cv2.resize(crop, None, fx=OLCEK, fy=OLCEK, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    yontemler = ["inv64", "clahe", "otsu"] if ESIKLEME == "coklu" else [ESIKLEME]
    return [tek_yontem(gray, y) for y in yontemler]


def araclari_takip_et(frame):
    """Aracları tespit edip [x1, y1, x2, y2, id] satırlarından oluşan dizi döndürür."""
    if TAKIPCI == "bytetrack":
        res = coco_model.track(frame, persist=True, tracker="bytetrack.yaml",
                               classes=VEHICLES, verbose=False)[0]
        if res.boxes is not None and res.boxes.id is not None:
            xyxy = res.boxes.xyxy.cpu().numpy()
            ids = res.boxes.id.cpu().numpy()
            return np.column_stack([xyxy, ids])
        return np.empty((0, 5))

    detections = coco_model(frame, verbose=False)[0]
    dets = [[x1, y1, x2, y2, s] for x1, y1, x2, y2, s, c in detections.boxes.data.tolist()
            if int(c) in VEHICLES]
    dets = np.asarray(dets) if dets else np.empty((0, 5))
    return mot_tracker.update(dets)


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

    # ---- araçları tespit et + takip et ----
    track_ids = araclari_takip_et(frame)

    # ---- plakaları tespit et ----
    license_plates = license_plate_detector(frame, verbose=False)[0]
    for license_plate in license_plates.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = license_plate

        # çok küçük/uzak plakaları okumaya çalışma (yanlış veri üretiyor)
        if (y2 - y1) < MIN_PLAKA_YUK or (x2 - x1) < MIN_PLAKA_GEN:
            continue

        # plakayı hangi araca ait olduğuna göre eşleştir
        xcar1, ycar1, xcar2, ycar2, car_id = get_car(license_plate, track_ids)

        if car_id != -1:                            # eşleşen araç bulunduysa
            license_plate_crop = frame[int(y1):int(y2), int(x1):int(x2), :]
            if license_plate_crop.size == 0:        # boş kırpmayı atla
                continue

            # her varyantı oku, en güvenli geçerli okumayı al
            en_iyi_text, en_iyi_skor = None, -1
            for varyant in plaka_varyantlari(license_plate_crop):
                text, text_score = read_license_plate(varyant)
                if text is not None and text_score > en_iyi_skor:
                    en_iyi_text, en_iyi_skor = text, text_score

            if en_iyi_text is None:
                continue

            # aynı karede aynı araca birden fazla plaka düşerse yüksek skorlu kalsın
            onceki = results[frame_nmr].get(car_id)
            if onceki and onceki['license_plate']['text_score'] >= en_iyi_skor:
                continue

            results[frame_nmr][car_id] = {
                'car': {'bbox': [xcar1, ycar1, xcar2, ycar2]},
                'license_plate': {'bbox': [x1, y1, x2, y2],
                                  'text': en_iyi_text,
                                  'bbox_score': score,
                                  'text_score': en_iyi_skor}}

    # ilerleme göstergesi
    if frame_nmr % 20 == 0:
        print(f"kare {frame_nmr} işlendi")

cap.release()

# sonuçları CSV'ye yaz
write_csv(results, './test.csv')
print("test.csv yazıldı")