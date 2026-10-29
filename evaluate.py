import json
import numpy as np, torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve
from model import AE, batch

ck = torch.load("model.pt")
T = ck["T"]
model = AE(T)
model.load_state_dict(ck["state"])
model.eval()

def scores(frames, vid):
    out = np.zeros(len(frames), dtype=np.float32)
    for v in np.unique(vid):
        idx = np.where(vid == v)[0]
        if len(idx) < T:
            continue
        f = frames[idx]
        starts = np.arange(len(f) - T + 1)
        errs = []
        with torch.no_grad():
            for i in range(0, len(starts), 256):
                x = batch(f, starts[i:i + 256], T)
                errs.append(((model(x) - x) ** 2).mean(dim=(1, 2, 3)).numpy())
        errs = np.concatenate(errs)
        out[idx[T - 1:]] = errs
        out[idx[:T - 1]] = errs[0]
    return out

# threshold from normal (training) data: mean + 3 std of reconstruction error
tf, tv = np.load("data_proc/train_frames.npy"), np.load("data_proc/train_vid.npy")
m = tv < 20
tr = scores(tf[m], tv[m])
thr = float(tr.mean() + 3 * tr.std())

# test scores + AUC (per-video min-max normalisation, the standard protocol)
xf, xv = np.load("data_proc/test_frames.npy"), np.load("data_proc/test_vid.npy")
y = np.load("data_proc/test_labels.npy")
s = scores(xf, xv)
sn = s.copy()
for v in np.unique(xv):
    k = xv == v
    sn[k] = (s[k] - s[k].min()) / (s[k].max() - s[k].min() + 1e-8)
auc = roc_auc_score(y, sn)
print(f"Frame-level AUC: {auc:.4f}")
print(f"Alert threshold (raw error): {thr:.6f}")
print(f"Test frames flagged at threshold: {(s > thr).mean() * 100:.1f}%")

json.dump({"threshold": thr, "T": T, "size": 64}, open("threshold.json", "w"))

# plots: ROC + score timeline of the first test video that has anomalies
fpr, tpr, _ = roc_curve(y, sn)
vids = [v for v in np.unique(xv) if y[xv == v].any()]
v = vids[0] if vids else np.unique(xv)[0]
k = xv == v
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
ax[0].plot(fpr, tpr)
ax[0].plot([0, 1], [0, 1], "--", color="gray")
ax[0].set(title=f"ROC curve (AUC = {auc:.3f})", xlabel="False positive rate", ylabel="True positive rate")
ax[1].plot(sn[k], label="anomaly score")
ax[1].fill_between(range(k.sum()), 0, 1, where=y[k] > 0, alpha=0.3, color="red", label="true anomaly")
ax[1].set(title=f"Test video {v}", xlabel="frame", ylabel="normalised score")
ax[1].legend()
plt.tight_layout()
plt.savefig("results.png", dpi=150)
print("Saved results.png and threshold.json")