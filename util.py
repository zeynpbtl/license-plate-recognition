import string
import easyocr
import re

# Initialize the OCR reader
reader = easyocr.Reader(['en'], gpu=False)

# OCR'ın okuyabileceği karakterler.
# O kaynaklı olarak plakada yok (0 ile karışmasın diye). Q, W, X Türk alfabesinde
# olmadığı için çıkarıldı (plakada yasak olduklarını doğrulayamadım, riski düşük).
# I bilerek dahil: yasak olduğuna dair kanıt bulamadım.
ALLOWLIST = 'ABCDEFGHIJKLMNPRSTUVYZ0123456789'

dict_char_to_int = {'O': '0', 'Q': '0', 'D': '0', 'I': '1', 'L': '1',
                    'Z': '2', 'S': '5', 'B': '8', 'G': '6', 'A': '4'}
dict_int_to_char = {'0': 'D', '1': 'I', '2': 'Z', '5': 'S',
                    '8': 'B', '6': 'G', '4': 'A'}
# NOT: '0' -> 'D' tahmindir (O plakada olmadığı için en benzer harf).
# I -> T gibi tahmini dönüşümler kaldırıldı.

# Türk plakası: il kodu (01-81) + 1-3 harf + 2-4 rakam
HARF = 'ABCDEFGHIJKLMNPRSTUVYZ'
PATTERN = re.compile(rf'(0[1-9]|[1-7][0-9]|8[01])[{HARF}]{{1,3}}\d{{2,4}}')


def write_csv(results, output_path):
    """
    Write the results to a CSV file.

    Args:
        results (dict): Dictionary containing the results.
        output_path (str): Path to the output CSV file.
    """
    with open(output_path, 'w') as f:
        f.write('{},{},{},{},{},{},{}\n'.format('frame_nmr', 'car_id', 'car_bbox',
                                                'license_plate_bbox', 'license_plate_bbox_score', 'license_number',
                                                'license_number_score'))

        for frame_nmr in results.keys():
            for car_id in results[frame_nmr].keys():
                print(results[frame_nmr][car_id])
                if 'car' in results[frame_nmr][car_id].keys() and \
                   'license_plate' in results[frame_nmr][car_id].keys() and \
                   'text' in results[frame_nmr][car_id]['license_plate'].keys():
                    f.write('{},{},{},{},{},{},{}\n'.format(frame_nmr,
                                                            car_id,
                                                            '[{} {} {} {}]'.format(
                                                                results[frame_nmr][car_id]['car']['bbox'][0],
                                                                results[frame_nmr][car_id]['car']['bbox'][1],
                                                                results[frame_nmr][car_id]['car']['bbox'][2],
                                                                results[frame_nmr][car_id]['car']['bbox'][3]),
                                                            '[{} {} {} {}]'.format(
                                                                results[frame_nmr][car_id]['license_plate']['bbox'][0],
                                                                results[frame_nmr][car_id]['license_plate']['bbox'][1],
                                                                results[frame_nmr][car_id]['license_plate']['bbox'][2],
                                                                results[frame_nmr][car_id]['license_plate']['bbox'][3]),
                                                            results[frame_nmr][car_id]['license_plate']['bbox_score'],
                                                            results[frame_nmr][car_id]['license_plate']['text'],
                                                            results[frame_nmr][car_id]['license_plate']['text_score'])
                            )
        f.close()


def license_complies_format(text):
    return format_license(text) is not None


def format_license(text):
    """Pozisyona göre düzeltir: ilk 2 rakam, sonra 1-3 harf, sonra rakamlar.
    Uygun biçim bulamazsa None döner."""
    if PATTERN.fullmatch(text):
        return text

    en_iyi, en_az = None, 99
    for k in (1, 2, 3):                      # harf sayısı
        if not 2 <= len(text) - 2 - k <= 4:  # sondaki rakam sayısı
            continue
        yeni, degisim = [], 0
        for i, c in enumerate(text):
            if 2 <= i < 2 + k:               # harf olmalı
                n = dict_int_to_char.get(c, c)
            else:                            # rakam olmalı
                n = dict_char_to_int.get(c, c)
            degisim += (n != c)
            yeni.append(n)
        aday = ''.join(yeni)
        if PATTERN.fullmatch(aday) and degisim < en_az:
            en_iyi, en_az = aday, degisim
    return en_iyi


def read_license_plate(license_plate_crop):
    """
    Read the Turkish license plate text from the given cropped image.

    Returns:
        tuple: formatted license plate text and its confidence score.
    """
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
        text = text.upper().replace(' ', '')
        text = re.sub(r'[^A-Z0-9]', '', text)
        if text.startswith('TR'):            # mavi şerit
            text = text[2:]

        if license_complies_format(text):
            return format_license(text), score

    return None, None


def get_car(license_plate, vehicle_track_ids):
    """
    Retrieve the vehicle coordinates and ID based on the license plate coordinates.

    Args:
        license_plate (tuple): Tuple containing the coordinates of the license plate (x1, y1, x2, y2, score, class_id).
        vehicle_track_ids (list): List of vehicle track IDs and their corresponding coordinates.

    Returns:
        tuple: Tuple containing the vehicle coordinates (x1, y1, x2, y2) and ID.
    """
    x1, y1, x2, y2, score, class_id = license_plate

    foundIt = False
    for j in range(len(vehicle_track_ids)):
        xcar1, ycar1, xcar2, ycar2, car_id = vehicle_track_ids[j]

        if x1 > xcar1 and y1 > ycar1 and x2 < xcar2 and y2 < ycar2:
            car_indx = j
            foundIt = True
            break

    if foundIt:
        return vehicle_track_ids[car_indx]

    return -1, -1, -1, -1, -1