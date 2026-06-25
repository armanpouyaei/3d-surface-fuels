"""Does a spatial CNN ('partial CNN') beat the v2 domain-adapted GBM at cross-ecosystem
transfer? Leave-one-site-out on the cached grids: a compact CNN on per-tile-standardized
AEF+S1 patches vs the GBM+tile-std baseline (LOSO mean R² ~0.01 / Spearman ~0.30). Honest
benchmark — only worth a v2.1 if the CNN materially wins. Runs on Apple GPU (MPS).

Usage:  python scripts/portable_v2_cnn.py
"""

import glob
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np  # noqa: E402
from numpy.lib.stride_tricks import sliding_window_view  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

from surface_fuels import metrics as M  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")
P = 7           # patch side (spatial context)
EPOCHS = 60


def load_grids():
    out = {}
    for f in sorted(glob.glob(os.path.join(ROOT, "data", "interim", "portable_grid_*.npz"))):
        site = os.path.basename(f).replace("portable_grid_", "").replace(".npz", "")
        d = np.load(f); out[site] = {k: d[k] for k in d.files}
    return out


def site_patches(g):
    """Per-tile-standardized AEF + raw S1 → (N,66,P,P) patches at valid cells + target."""
    aef = np.nan_to_num(g["aef"]).astype(np.float32)
    aef = (aef - aef.reshape(64, -1).mean(1)[:, None, None]) / (aef.reshape(64, -1).std(1)[:, None, None] + 1e-6)
    stack = np.concatenate([aef, np.nan_to_num(g["s1"]).astype(np.float32)], 0)  # (66,ny,nx)
    pad = P // 2
    padded = np.pad(stack, ((0, 0), (pad, pad), (pad, pad)), mode="reflect")
    sw = sliding_window_view(padded, (P, P), axis=(1, 2))           # (66,ny,nx,P,P)
    patches = np.ascontiguousarray(np.moveaxis(sw, 0, 2)).reshape(-1, stack.shape[0], P, P)
    valid = g["valid"].ravel()
    return patches[valid].astype(np.float32), g["target"].ravel()[valid].astype(np.float32)


def build_cnn(cin):
    import torch.nn as nn
    return nn.Sequential(
        nn.Conv2d(cin, 48, 3, padding=1, bias=False), nn.BatchNorm2d(48), nn.SiLU(),
        nn.Conv2d(48, 48, 3, padding=1, groups=48, bias=False), nn.BatchNorm2d(48), nn.SiLU(),
        nn.Conv2d(48, 64, 1, bias=False), nn.BatchNorm2d(64), nn.SiLU(),
        nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.1), nn.Linear(64, 1))


def train_predict(Xtr, ytr, Xte):
    import torch
    torch.manual_seed(0)
    dev = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    net = build_cnn(Xtr.shape[1]).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=2e-3, weight_decay=1e-4)
    Xt = torch.tensor(Xtr, device=dev); yt = torch.tensor(ytr, device=dev).view(-1, 1)
    n, bs = len(Xt), 256
    net.train()
    for _ in range(EPOCHS):
        perm = torch.randperm(n, device=dev)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            loss = ((net(Xt[idx]) - yt[idx]) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
    net.eval()
    with torch.no_grad():
        out = []
        Xe = torch.tensor(Xte, device=dev)
        for i in range(0, len(Xe), 4096):
            out.append(net(Xe[i:i + 4096]).cpu().numpy().ravel())
    return np.clip(np.concatenate(out), 0, None)


def main():
    import torch  # noqa: F401
    grids = load_grids()
    if len(grids) < 2:
        sys.exit("Need cached grids — run scripts/train_portable.py first.")
    feats = {n: site_patches(grids[n]) for n in grids}
    names = list(grids)
    print(f"CNN LOSO (patch {P}×{P}, per-tile-std AEF+S1) on {names}\n")
    r2s, sps = [], []
    for held in names:
        Xtr = np.concatenate([feats[n][0] for n in names if n != held])
        ytr = np.concatenate([feats[n][1] for n in names if n != held])
        Xte, yte = feats[held]
        pred = train_predict(Xtr, ytr, Xte)
        r2 = M.r2(pred, yte); sp = spearmanr(pred, yte).correlation
        r2s.append(r2); sps.append(sp)
        print(f"  → {held:5}: R² {r2:+.3f}  Spearman {sp:+.3f}  (n_test {len(yte)})", flush=True)
    print(f"\n  CNN MEAN: R² {np.mean(r2s):+.3f}  Spearman {np.mean(sps):+.3f}")
    print("  vs GBM+tile-std (v2): R² +0.008  Spearman +0.303")
    print("  → CNN is worth a v2.1 only if it materially beats those.")


if __name__ == "__main__":
    main()
