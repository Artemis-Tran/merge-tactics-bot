#!/usr/bin/env python3
import json
import argparse
import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Any
from PIL import Image

import pytesseract

from adb_wrap import adb_swipe, adb_screenshot

GEOM_PATH = Path("geometry.json")
TESS_CFG = "--psm 7 -c tessedit_char_whitelist=0123456789"
OUT_DIR = Path("runs"); OUT_DIR.mkdir(exist_ok=True)
ROI_DIR = OUT_DIR / "rois"; ROI_DIR.mkdir(exist_ok=True)

# helpers: normalized → pixels
def norm_to_px(val: float, total: int) -> int:
    return int(round(val * total))

def norm_rect_to_px(r: Dict[str, float], W: int, H: int) -> Tuple[int,int,int,int]:
    return (norm_to_px(r["x"], W), norm_to_px(r["y"], H), norm_to_px(r["w"], W), norm_to_px(r["h"], H))

def center_wh_to_rect_px(cx: float, cy: float, w: float, h: float, W: int, H: int) -> Tuple[int,int,int,int]:
    px_w = norm_to_px(w, W)
    px_h = norm_to_px(h, H)
    px_cx = norm_to_px(cx, W)
    px_cy = norm_to_px(cy, H)
    return (px_cx - px_w // 2, px_cy - px_h // 2, px_w, px_h)

def rect_center(rect: Tuple[int,int,int,int]) -> Tuple[int,int]:
    x, y, w, h = rect
    return (x + w // 2, y + h // 2)

def load_geom(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text())

def crop(img: np.ndarray, r: Tuple[int,int,int,int]) -> np.ndarray:
    x,y,w,h = r; return img[y:y+h, x:x+w].copy()

def draw_rect(img, r, color, label=None, thick=2):
    x,y,w,h = r
    cv2.rectangle(img, (x,y), (x+w, y+h), color, thick)
    if label:
        cv2.putText(img, label, (x, y-6), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)

def ocr_digits(img_bgr: np.ndarray) -> int | None:
    if img_bgr.size == 0: return None
    g = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    g = cv2.resize(g, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_LINEAR)
    # a few simple variants; return first that yields digits
    _, th_bin = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    _, th_inv = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    variants = [th_bin, cv2.medianBlur(th_bin,3), th_inv, cv2.medianBlur(th_inv,3)]
    for v in variants:
        txt = pytesseract.image_to_string(Image.fromarray(v), config=TESS_CFG)
        digits = "".join(ch for ch in txt if ch.isdigit())
        if digits:
            try: return int(digits)
            except: pass
    return None

def main():
    parser = argparse.ArgumentParser(description="Bare-bones drag: hand -> tile")
    parser.add_argument("--hand", type=int, default=0, help="hand slot index (0-based)")
    parser.add_argument("--board", type=int, nargs=2, metavar=("ROW", "COL"),
                    default=[2, 2],
                    help="target board position as two ints: ROW COL (0-based)")
    parser.add_argument("--duration", type=int, default=100, help="swipe duration ms")
    parser.add_argument("--dry", action="store_true", help="print coords but don't send swipe")
    parser.add_argument("--ocr", action="store_true", help="read mana + hand costs from screenshot before drag")
    args = parser.parse_args()

    geom = load_geom(GEOM_PATH)
    W, H = geom["resolution_px"]

    # compute all tile rects once
    rows = geom["board"]["rows"]
    cols = geom["board"]["cols"]
    tiles = [
        center_wh_to_rect_px(t["cx"], t["cy"], t["w"], t["h"], W, H)
        for t in geom["board"]["tiles"]
    ]

    # target tile rect
    if not (0 <= args.board[0] < rows and 0 <= args.board[1] < cols):
        raise SystemExit(f"row/col out of range. rows={rows}, cols={cols}")
    tile_rect = tiles[args.board[0] * cols + args.board[1]]
    tx, ty = rect_center(tile_rect)

    # hand rect
    if not (0 <= args.hand < len(geom["hand"])):
        raise SystemExit(f"hand index out of range. have {len(geom['hand'])} hand slots")
    h = geom["hand"][args.hand]
    hand_rect = center_wh_to_rect_px(h["cx"], h["cy"], h["w"], h["h"], W, H)
    hx, hy = rect_center(hand_rect)

    mana_val = None
    costs: List[int | None] = []
    if args.ocr:
        # 1) screenshot
        scr_path = OUT_DIR / "latest.png"
        adb_screenshot(scr_path)
        img = cv2.imread(str(scr_path), cv2.IMREAD_COLOR)
        if img is None: raise SystemExit("Failed to read screenshot")

        # 2) build all hand rects + cost ROIs
        hand_rects_all: List[Tuple[int,int,int,int]] = []
        cost_rois: List[Tuple[int,int,int,int]] = []
        for hh in geom["hand"]:
            hr = center_wh_to_rect_px(hh["cx"], hh["cy"], hh["w"], hh["h"], W, H)
            hand_rects_all.append(hr)
            rx, ry, rw, rh = hr
            rel = hh["cost_roi_rel"]
            cr = (rx + int(round(rel["x"]*rw)),
                  ry + int(round(rel["y"]*rh)),
                  int(round(rel["w"]*rw)),
                  int(round(rel["h"]*rh)))
            cost_rois.append(cr)

        mana_rect = norm_rect_to_px(geom["mana_roi"], W, H)

        # 3) OCR costs + mana
        for i, roi in enumerate(cost_rois):
            patch = crop(img, roi)
            cv2.imwrite(str(ROI_DIR / f"hand_cost_{i}.png"), patch)
            costs.append(ocr_digits(patch))

        mana_img = crop(img, mana_rect)
        cv2.imwrite(str(ROI_DIR / "mana.png"), mana_img)
        mana_val = ocr_digits(mana_img)

        print(f"[ocr] mana={mana_val}  hand_costs={costs}")

        # 4) overlay (optional)
        
        dbg = img.copy()
        draw_rect(dbg, mana_rect, (0,255,255), label=f"mana: {mana_val if mana_val is not None else '?'}")
        for i, hr in enumerate(hand_rects_all):
            draw_rect(dbg, hr, (200,0,200), label=f"hand {i}")
            draw_rect(dbg, cost_rois[i], (140,0,140), label=f"${costs[i] if costs[i] is not None else '?'}")
        out_overlay = OUT_DIR / "ocr_overlay.png"
        cv2.imwrite(str(out_overlay), dbg)
        print(f"[ok] wrote {out_overlay}")

    print(f"[info] dragging hand[{args.hand}] ({hx},{hy}) -> tile({args.board[0]},{args.board[1]}) center ({tx},{ty}), duration={args.duration}ms")

    if args.dry:
        print("[dry-run] not sending swipe (use without --dry to execute).")
        return

    # do the swipe
    adb_swipe(hx, hy, tx, ty, args.duration)
    print("[ok] swipe sent")

if __name__ == "__main__":
    main()
