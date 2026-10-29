import argparse, time
import numpy as np, torch
from model import AE, batch

ap = argparse.ArgumentParser()
ap.add_argument("--epochs", type=int, default=8)
ap.add_argument("--bs", type=int, default=128)
ap.add_argument("--T", type=int, default=4)
args = ap.parse_args()
T = args.T

frames = np.load("data_proc/train_frames.npy")
vid = np.load("data_proc/train_vid.npy")
starts = np.where(vid[:len(vid) - T + 1] == vid[T - 1:])[0]   # windows inside one video
print(f"{len(frames)} frames, {len(starts)} training windows")

model = AE(T)
opt = torch.optim.Adam(model.parameters(), 1e-3)
loss_fn = torch.nn.MSELoss()
for ep in range(args.epochs):
    t0, tot, n = time.time(), 0.0, 0
    np.random.shuffle(starts)
    for i in range(0, len(starts), args.bs):
        x = batch(frames, starts[i:i + args.bs], T)
        loss = loss_fn(model(x), x)
        opt.zero_grad(); loss.backward(); opt.step()
        tot += loss.item() * len(x); n += len(x)
    print(f"epoch {ep + 1}/{args.epochs}  loss {tot / n:.5f}  ({time.time() - t0:.0f}s)", flush=True)
    torch.save({"state": model.state_dict(), "T": T}, "model.pt")