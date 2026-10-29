import numpy as np, torch, torch.nn as nn

class AE(nn.Module):
    def __init__(self, T=4):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(T, 32, 3, 2, 1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, 2, 1), nn.ReLU(),
            nn.Conv2d(64, 128, 3, 2, 1), nn.ReLU())
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(32, T, 4, 2, 1), nn.Sigmoid())
    def forward(self, x):
        return self.dec(self.enc(x))

def batch(frames, idx, T):
    """idx = start indices -> tensor (B, T, H, W) in [0,1]"""
    return torch.from_numpy(frames[idx[:, None] + np.arange(T)]).float() / 255.0