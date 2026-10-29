import argparse
from pathlib import Path
import numpy as np, cv2

ap = argparse.ArgumentParser()
ap.add_argument("--root", required=True)            # the extracted folder
ap.add_argument("--out", default="data_proc")
ap.add_argument("--size", type=int, default=64)
ap.add_argument("--stride", type=int, default=2)    # keep every 2nd frame
ap.add_argument("--max-train-videos", type=int, default=80)
ap.add_argument("--max-test-videos", type=int, default=40)
ap.add_argument("--max-train-frames", type=int, default=200)  # per video, after stride
args = ap.parse_args()
VID, IMG = {".avi", ".mp4"}, {".jpg", ".jpeg", ".png"}
root, out = Path(args.root), Path(args.out)
out.mkdir(exist_ok=True)

def units(split):
    vids = sorted(p for p in split.rglob("*") if p.suffix.lower() in VID)
    if vids:
        return vids
    return sorted(d for d in split.rglob("*") if d.is_dir()
                  and any(f.suffix.lower() in IMG for f in d.iterdir()))

def pick(lst, n):
    if len(lst) <= n:
        return lst
    return [lst[i] for i in np.linspace(0, len(lst) - 1, n).astype(int)]

def load(unit, maxf):
    fr = []
    def add(im):
        fr.append(cv2.resize(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY) if im.ndim == 3 else im,
                             (args.size, args.size)))
    if unit.is_dir():
        files = sorted(f for f in unit.iterdir() if f.suffix.lower() in IMG)
        for f in files[::args.stride][:maxf]:
            add(cv2.imread(str(f)))
    else:
        cap, i = cv2.VideoCapture(str(unit)), 0
        while len(fr) < maxf:
            ok, im = cap.read()
            if not ok:
                break
            if i % args.stride == 0:
                add(im)
            i += 1
        cap.release()
    return np.array(fr, dtype=np.uint8)

def find_split(names):
    for p in sorted(root.rglob("*")):
        if p.is_dir() and p.name.lower() in names:
            u = units(p)
            if u:
                return u
    return []

train_units = pick(find_split({"training", "train"}), args.max_train_videos)
test_units = find_split({"testing", "test"})
masks = {p.stem: p for p in root.rglob("*.npy") if "frame_mask" in str(p.parent).lower()}
if not train_units or not test_units:
    raise SystemExit(f"Could not find data: train={len(train_units)} test={len(test_units)}. Paste me the folder listing.")
print(f"{len(train_units)} train units, {len(test_units)} test units, {len(masks)} masks")

X, V = [], []
for i, u in enumerate(train_units):
    f = load(u, args.max_train_frames)
    X.append(f); V.append(np.full(len(f), i))
np.save(out / "train_frames.npy", np.concatenate(X))
np.save(out / "train_vid.npy", np.concatenate(V))

X, V, L = [], [], []
test_units = [u for u in test_units if (u.stem if u.is_file() else u.name) in masks]
for i, u in enumerate(pick(test_units, args.max_test_videos)):
    f = load(u, 10**9)
    lab = np.load(masks[u.stem if u.is_file() else u.name])[::args.stride][:len(f)]
    f = f[:len(lab)]
    X.append(f); V.append(np.full(len(f), i)); L.append(lab.astype(np.uint8))
np.save(out / "test_frames.npy", np.concatenate(X))
np.save(out / "test_vid.npy", np.concatenate(V))
np.save(out / "test_labels.npy", np.concatenate(L))
print("train frames:", sum(map(len, X)) and np.load(out / "train_frames.npy").shape,
      "| test frames:", np.load(out / "test_frames.npy").shape,
      "| anomalous test frames:", int(np.concatenate(L).sum()))