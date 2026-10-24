from ultralytics import YOLO

model = YOLO("best.pt")
model.predict("video.mp4", save=True, conf=0.25)