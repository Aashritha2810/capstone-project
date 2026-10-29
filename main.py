import argparse, json, sqlite3, time
from collections import deque
from datetime import datetime
import cv2, numpy as np, torch
from ultralytics import YOLO
from model import AE

ap = argparse.ArgumentParser()
ap.add_argument("--source", default="0")          # 0 = webcam, or a video file path
ap.add_argument("--camera-id", default="cam_01")
ap.add_argument("--db", default="surveillance.db")
ap.add_argument("--calib-seconds", type=float, default=10)  # webcam only
ap.add_argument("--stride", type=int, default=1)  # use 2 for dataset videos (matches training)
args = ap.parse_args()
is_cam = args.source.isdigit()
source = int(args.source) if is_cam else args.source

ck = torch.load("model.pt")
T = ck["T"]
ae = AE(T)
ae.load_state_dict(ck["state"])
ae.eval()
cfg = json.load(open("threshold.json"))
size, thr = cfg["size"], cfg["threshold"]

db = sqlite3.connect(args.db)
db.execute("""CREATE TABLE IF NOT EXISTS detections (
    id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp TEXT, camera_source TEXT,
    track_id INTEGER, object_type TEXT, confidence REAL,
    x1 INTEGER, y1 INTEGER, x2 INTEGER, y2 INTEGER)""")
db.execute("""CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp TEXT, camera_source TEXT,
    anomaly_score REAL, threshold REAL, alert INTEGER)""")

yolo = YOLO("yolo11n.pt")
buf, recent, calib = deque(maxlen=T), deque(maxlen=5), []
t0, n, score, alert = time.time(), 0, 0.0, False

for r in yolo.track(source=source, stream=True, persist=True, imgsz=480, conf=0.4, verbose=False):
    n += 1
    ts = datetime.now().isoformat(timespec="milliseconds")
    img = r.orig_img

    # anomaly score
    if n % args.stride == 0:
        g = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (size, size))
        buf.append(g)
        if len(buf) == T:
            x = torch.from_numpy(np.stack(buf)).float().unsqueeze(0) / 255.0
            with torch.no_grad():
                err = float(((ae(x) - x) ** 2).mean())
            recent.append(err)
            score = float(np.mean(recent))
            if is_cam and time.time() - t0 < args.calib_seconds:
                calib.append(score)
    calibrating = is_cam and time.time() - t0 < args.calib_seconds
    if is_cam and not calibrating and calib:
        thr = float(np.mean(calib) + 3 * np.std(calib))
        calib = []