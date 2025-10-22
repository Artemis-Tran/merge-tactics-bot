#!/usr/bin/env python3
# crop_mana_rois.py
from pathlib import Path
import json
import cv2

GEOM_PATH = Path("../geometry.json")
SRC_DIR   = Path("../assets/uncropped")
OUT_DIR   = Path("../assets/mana_rois")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def norm_to_abs_rect(nr, W, H):
    x = max(0, int(round(nr["x"] * W)))
    y = max(0, int(round(nr["y"] * H)))
    w = max(1, int(round(nr["w"] * W)))
    h = max(1, int(round(nr["h"] * H)))
    # clamp to image bounds
    x = min(x, W - 1)
    y = min(y, H - 1)
    w = min(w, W - x)
    h = min(h, H - y)
    return x, y, w, h

def main():
    geom = json.loads(GEOM_PATH.read_text())
    mana_roi_norm = geom["mana_roi"]  # expects keys: x,y,w,h in [0,1]

    imgs = sorted([p for p in SRC_DIR.iterdir() if p.suffix.lower() in {".png",".jpg",".jpeg"}])
    if not imgs:
        print(f"No images found in {SRC_DIR}")
        return

    for p in imgs:
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img is None:
            print(f"Skip unreadable: {p}")
            continue

        H, W = img.shape[:2]
        x, y, w, h = norm_to_abs_rect(mana_roi_norm, W, H)
        crop = img[y:y+h, x:x+w]

        out = OUT_DIR / f"{p.stem}_mana.png"
        cv2.imwrite(str(out), crop)
        print(f"Cropped {p.name} -> {out.name}  ({W}x{H} -> {w}x{h})")

    print(f"Done. Crops saved in: {OUT_DIR.resolve()}")

if __name__ == "__main__":
    main()
