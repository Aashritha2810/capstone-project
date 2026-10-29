import argparse, sqlite3
from datetime import datetime
import cv2
from ultralytics import YOLO

ap = argparse.ArgumentParser()
ap.add_argument("--source", default="0")        # 0 = webcam, or a video file path
ap.add_argument("--camera-id", default="cam_01")
ap.add_argument("--db", default="surveillance.db")
args = ap.parse_args()
source = int(args.source) if args.source.isdigit() else args.source

db = sqlite3.connect(args.db)
db.execute("""CREATE TABLE IF NOT EXISTS detections (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT, camera_source TEXT, track_id INTEGER,
    object_type TEXT, confidence REAL,
    x1 INTEGER, y1 INTEGER, x2 INTEGER, y2 INTEGER)""")

model = YOLO("yolo11n.pt")  # small model, downloads automatically, OK on CPU

for r in model.track(source=source, stream=True, persist=True,
                     imgsz=480, conf=0.4, verbose=False):
    ts = datetime.now().isoformat(timespec="milliseconds")
    if r.boxes is not None and len(r.boxes) > 0:
        ids = r.boxes.id.int().tolist() if r.boxes.id is not None else [None] * len(r.boxes)
        for box, tid in zip(r.boxes, ids):
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            db.execute(
                "INSERT INTO detections (timestamp, camera_source, track_id, object_type,"
                " confidence, x1, y1, x2, y2) VALUES (?,?,?,?,?,?,?,?,?)",
                (ts, args.camera_id, tid, model.names[int(box.cls)],
                 float(box.conf), x1, y1, x2, y2))
        db.commit()
    cv2.imshow("Surveillance", r.plot())
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cv2.destroyAllWindows()
db.close()