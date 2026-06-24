"""#2: GEDI L2B understory features over OSBS.

Extracts footprint-level PAVD (Plant Area Volume Density) vertical-profile bins —
0-5 m and 5-10 m, the near-ground/understory signal AlphaEarth & Sentinel-1 lack —
plus PAI, cover, RH100, from downloaded GEDI02_B granules. Quantifies coverage at
the OSBS AOI scale and shows the understory structure.

Usage: python scripts/build_gedi_osbs.py
"""

import glob
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import h5py  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")
BOX = (-82.05, 29.64, -81.94, 29.74)        # OSBS region
TIGHT = (-81.998, 29.685, -81.988, 29.694)  # NEON clip AOI (1 km)
BEAMS = ["BEAM0000", "BEAM0001", "BEAM0010", "BEAM0011",
         "BEAM0101", "BEAM0110", "BEAM1000", "BEAM1011"]


def extract(path):
    rows = []
    prof = []
    with h5py.File(path, "r") as f:
        for b in BEAMS:
            if b not in f:
                continue
            g = f[b]
            lat = g["geolocation/lat_lowestmode"][:]
            lon = g["geolocation/lon_lowestmode"][:]
            q = g["l2b_quality_flag"][:]
            pavd = g["pavd_z"][:]            # (n, 30) 5 m bins
            pai = g["pai"][:]; cover = g["cover"][:]; rh100 = g["rh100"][:]
            m = (lon >= BOX[0]) & (lon <= BOX[2]) & (lat >= BOX[1]) & (lat <= BOX[3]) & (q == 1)
            for i in np.where(m)[0]:
                rows.append({"lon": lon[i], "lat": lat[i], "pavd_0_5": pavd[i, 0],
                             "pavd_5_10": pavd[i, 1], "pai": pai[i], "cover": cover[i],
                             "rh100": rh100[i] / 100.0})
                prof.append(pavd[i, :12])    # 0-60 m
    return rows, prof


def main():
    gran = sorted(glob.glob(os.path.join(ROOT, "data", "raw", "gedi", "*.h5")))
    if not gran:
        sys.exit("No GEDI granules — run the download step first.")
    rows, prof = [], []
    for p in gran:
        r, pr = extract(p)
        rows += r; prof += pr
        print(f"  {os.path.basename(p)[:38]}: {len(r)} quality footprints in OSBS box")
    d = pd.DataFrame(rows)
    if d.empty:
        sys.exit("No footprints in box.")
    tight = d[(d.lon >= TIGHT[0]) & (d.lon <= TIGHT[2]) & (d.lat >= TIGHT[1]) & (d.lat <= TIGHT[3])]
    print(f"\n{len(d)} GEDI footprints over OSBS box ({len(gran)} granules); "
          f"{len(tight)} in the tight 1 km NEON clip AOI")
    print(f"understory PAVD 0-5 m: mean {d.pavd_0_5.mean():.3f} m²/m³ | "
          f"5-10 m: {d.pavd_5_10.mean():.3f} | PAI {d.pai.mean():.2f} | cover {d.cover.mean():.2f}")

    prof = np.array(prof)
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
    sc = ax[0].scatter(d.lon, d.lat, c=d.pavd_0_5, cmap="YlGn", s=14, vmin=0,
                       vmax=np.nanpercentile(d.pavd_0_5, 95))
    ax[0].add_patch(plt.Rectangle((TIGHT[0], TIGHT[1]), TIGHT[2]-TIGHT[0], TIGHT[3]-TIGHT[1],
                                  fill=False, ec="r", lw=1.5))
    ax[0].set_title(f"GEDI footprints over OSBS (n={len(d)}; red=clip AOI, n={len(tight)})")
    ax[0].set_xlabel("lon"); ax[0].set_ylabel("lat"); plt.colorbar(sc, ax=ax[0], label="0-5 m PAVD")
    z = np.arange(12) * 5 + 2.5
    idx = np.argsort(d.pai.values)[::max(1, len(d)//8)][:8]
    for i in idx:
        ax[1].plot(prof[i], z, alpha=.6)
    ax[1].axhspan(0, 10, color="orange", alpha=.12, label="understory 0-10 m")
    ax[1].set_xlabel("PAVD (m²/m³)"); ax[1].set_ylabel("height (m)")
    ax[1].set_title("Example GEDI vertical PAVD profiles"); ax[1].legend(fontsize=8)
    ax[2].hist(d.pavd_0_5, bins=25, color="#31a354", edgecolor="w")
    ax[2].set_xlabel("understory PAVD 0-5 m (m²/m³)"); ax[2].set_ylabel("# footprints")
    ax[2].set_title("Understory density distribution")
    plt.tight_layout()
    fp = os.path.join(ROOT, "figures", "gedi_osbs.png")
    os.makedirs(os.path.dirname(fp), exist_ok=True); plt.savefig(fp, dpi=110)
    d.to_csv(os.path.join(ROOT, "data", "interim", "gedi_osbs.csv"), index=False)
    print(f"saved {fp} and data/interim/gedi_osbs.csv")


if __name__ == "__main__":
    main()
