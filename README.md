# Türk Plaka Tanıma (ANPR) – Video Üzerinde

Videodaki araçları tespit edip takip eden, plakalarını okuyan ve sonucu videonun üzerine çizen bir Python projesidir. Türk plaka formatına (il kodu + 1-3 harf + rakamlar) göre okumaları doğrular ve düzeltir.

> **Durum:** Deneysel / demo seviyesi. (bkz. [Sınırlamalar](#sınırlamalar)).

## Nasıl çalışır?

```
video ──► araç tespiti (YOLO) ──► takip (ByteTrack) ──┐
      └─► plaka tespiti (özel YOLO modeli) ───────────┤
                                                      ▼
                          plakayı araca eşle (en küçük kapsayan kutu)
                                                      ▼
              kırp → büyüt → gri → eşikle (inv64 / CLAHE / Otsu)
                                                      ▼
                       EasyOCR → Türk plaka formatı doğrulama/düzeltme
                                                      ▼
                                     test.csv (kare kare okumalar)
                                                      ▼
            add_missing_data.py: eksik kareleri interpolasyonla doldur,
                       ardışık ID'leri birleştir → test_interpolated.csv
                                                      ▼
            visualize.py: araç başına karakter bazlı ağırlıklı oylama,
                   plaka benzerliğine göre ID birleştirme → out.mp4
```

Öne çıkan noktalar:

- **Çoklu önişleme:** Her plaka üç farklı önişlemeyle (inv64, CLAHE, Otsu) okunur, en güvenli geçerli okuma seçilir.
- **Format doğrulama:** İl kodu 01-81, harf sayısına göre rakam sayısı kontrolü; yalnızca en fazla 1 karakterlik OCR düzeltmesi kabul edilir.
- **Karakter bazlı oylama:** Bir aracın tüm okumaları harf/rakam yapısına göre gruplanır, her pozisyon `skor³` ağırlığıyla oylanır. İl kodu tek parça oylanır.
- **ID birleştirme:** Takipçi aynı araca farklı ID verirse, hem konuma (IoU) hem plaka benzerliğine göre birleştirilir.

## Kurulum

Python 3.10+ önerilir.

```bash
git clone https://github.com/zeynpbtl/license-plate-recognition.git
cd license-plate-recognition
python -m venv .venv
.venv\Scripts\activate        # Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

### Gerekli dosyalar (repoda yok)

| Dosya | Açıklama |
|---|---|
| `video_h264.mp4` | İşlenecek video (`main.py` ve `visualize.py` içindeki yolla aynı olmalı) |
| `best.pt` | Plaka tespiti için eğitilmiş YOLO modeli |
| `yolo26n.pt` | Araç tespiti modeli (Ultralytics ilk çalıştırmada kendisi indirir) |

## Kullanım

Üç adım, sırayla çalıştırılır:

```bash
python main.py               # tespit + takip + OCR  -> test.csv
python add_missing_data.py   # interpolasyon + ID birleştirme -> test_interpolated.csv
python visualize.py          # oylama + çizim -> out.mp4
```

Hızlı deneme için `main.py` içinde `MAX_FRAME = 100` yapabilirsin.

## Ayarlar

Tüm ayarlar ilgili dosyaların başındaki sabitlerdedir.

| Dosya | Ayar | Anlamı |
|---|---|---|
| `main.py` | `ESIKLEME` | `"coklu"` (yavaş, daha iyi), `"inv64"`, `"otsu"`, `"clahe"`, `"adaptif"`, `"yok"` |
| `main.py` | `TAKIPCI` | `"bytetrack"` veya `"sort"` |
| `main.py` | `MIN_PLAKA_YUK`, `MIN_PLAKA_GEN` | Bundan küçük (uzak) plakalar okunmaz |
| `util.py` | `MAX_DEGISIM` | Format düzeltmesinde değiştirilebilecek en fazla karakter |
| `util.py` | `RAKAM_ARALIK` | Harf sayısına göre izin verilen rakam sayısı |
| `add_missing_data.py` | `MAX_BOSLUK`, `MIN_IOU` | Konuma göre ID birleştirme eşikleri |
| `visualize.py` | `MIN_SKOR`, `USS`, `MIN_OKUMA` | Oylama eşikleri |
| `visualize.py` | `MAX_FARK` | Plaka benzerliğine göre ID birleştirme toleransı |
| `visualize.py` | `GERCEK` | `{araç_id: "plaka"}` yazarsan doğruluk yüzdesi basılır |

## Doğruluğu ölçmek

1. `visualize.py`'yi çalıştır, terminalde araç ID'lerini gör.
2. Gerçek plakaları `GERCEK = {1: "34GT2850", ...}` olarak yaz.
3. Her ayar değişikliğinden sonra doğruluk yüzdesini karşılaştır.

## Sınırlamalar

- Tek bir trafik videosunda geliştirildi, farklı ışık/açı/çözünürlükte sonuçlar değişir.
- Uzak ve hareket bulanıklığı olan plakalarda EasyOCR benzer harfleri karıştırabiliyor (örn. M/N, 3/0).
- Plakaları birbirine çok benzeyen iki farklı araç art arda geçerse ID birleştirme yanlışlıkla birleştirebilir (`MAX_FARK` ile ayarlanır).
- `RAKAM_ARALIK` içindeki harf/rakam kuralları gözlemlere dayanıyor, resmi yönetmelikle doğrulanmadı.
- Gerçek zamanlı değil; çevrimdışı video işleme için tasarlandı.

## Fikirler / yapılacaklar

- [ ] Komut satırı argümanları (video yolu, çıktı yolu)
- [ ] PaddleOCR karşılaştırması
- [ ] Plaka karakterleri için özel eğitilmiş tespit modeli
- [ ] Çok kareli süper çözünürlük
- [ ] Etiketli bir değerlendirme seti

## Gizlilik ve telif notu

Plakalar kişisel veri sayılabilir. Repoya video ve model dosyaları eklenmemiştir.

## Kullanılan açık kaynak projeler

- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) (AGPL-3.0)
- [EasyOCR](https://github.com/JaidedAI/EasyOCR) (Apache-2.0)
- [OpenCV](https://opencv.org/), NumPy, pandas
- (İsteğe bağlı) [SORT](https://github.com/abewley/sort)

## Lisans

Ultralytics AGPL-3.0 lisanslı olduğu için bu projeyi de AGPL-3.0 altında yayınlamak en güvenli seçenektir. Detay için `LICENSE` dosyasına bak.
