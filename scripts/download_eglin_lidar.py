"""Download real USGS 3DEP LiDAR over Eglin AFB (Okaloosa County, FL).

Uses The National Map's ungated TNM Access API to find Lidar Point Cloud tiles
for a bbox, then downloads the smallest into data/raw/. This is the real-data
on-ramp for the Eglin surface-fuel demo (Approach A).

Note on RxCADRE TLS: the RxCADRE 2012 terrestrial scans (DOI 10.2737/RDS-2023-0011)
and ground-fuel clip plots (RDS-2014-0031) are the ideal co-located input+truth
pair, but their hosts (AgData Commons / FS RDS) are bot-gated and not reliably
scriptable. Download them manually from the catalog pages into data/raw/ to plug
into the same voxelizer; USGS 3DEP below is the scriptable real-LiDAR stand-in.

Usage:  python scripts/download_eglin_lidar.py
"""

import os
import sys
import urllib.request

ROOT = os.path.join(os.path.dirname(__file__), "..")
RAW = os.path.join(ROOT, "data", "raw")
UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/120 Safari/537.36"

# Eglin HIP area bbox (lon/lat) and the verified smallest tile.
TNM = ("https://tnmaccess.nationalmap.gov/api/v1/products"
       "?datasets=Lidar%20Point%20Cloud%20(LPC)&bbox=-86.80,30.45,-86.60,30.60"
       "&prodFormats=LAS,LAZ&max=15")
DEFAULT_TILE = ("https://rockyweb.usgs.gov/vdelivery/Datasets/Staged/Elevation/LPC/"
                "Projects/legacy/FL_OKALOOSACO_2007/LAZ/"
                "USGS_LPC_FL_OKALOOSACO_2007_000051.laz")
DEST = os.path.join(RAW, "eglin_3dep_000051.laz")


def _get(url):
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    return urllib.request.urlopen(req, timeout=120)


def main():
    os.makedirs(RAW, exist_ok=True)
    if os.path.exists(DEST) and os.path.getsize(DEST) > 1_000_000:
        print(f"Already have {DEST} ({os.path.getsize(DEST)/1e6:.1f} MB)")
        return
    print(f"Downloading {DEFAULT_TILE.split('/')[-1]} ...")
    with _get(DEFAULT_TILE) as r, open(DEST, "wb") as fh:
        fh.write(r.read())
    print(f"Saved {DEST} ({os.path.getsize(DEST)/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
