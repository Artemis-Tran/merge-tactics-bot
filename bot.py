#!/usr/bin/env python3
import json
import argparse
from pathlib import Path
from typing import Tuple, Dict, Any

# uses your working ADB wrapper
from adb_wrap import adb_swipe

GEOM_PATH = Path("geometry.json")

# helpers: normalized → pixels
def norm_to_px(val: float, total: int) -> int:
    return int(round(val * total))

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

def main():
    parser = argparse.ArgumentParser(description="Bare-bones drag: hand -> tile")
    parser.add_argument("--hand", type=int, default=0, help="hand slot index (0-based)")
    parser.add_argument("--row",  type=int, default=2, help="board row (0-based)")
    parser.add_argument("--col",  type=int, default=2, help="board col (0-based)")
    parser.add_argument("--duration", type=int, default=280, help="swipe duration ms")
    parser.add_argument("--dry", action="store_true", help="print coords but don't send swipe")
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
    if not (0 <= args.row < rows and 0 <= args.col < cols):
        raise SystemExit(f"row/col out of range. rows={rows}, cols={cols}")
    tile_rect = tiles[args.row * cols + args.col]
    tx, ty = rect_center(tile_rect)

    # hand rect
    if not (0 <= args.hand < len(geom["hand"])):
        raise SystemExit(f"hand index out of range. have {len(geom['hand'])} hand slots")
    h = geom["hand"][args.hand]
    hand_rect = center_wh_to_rect_px(h["cx"], h["cy"], h["w"], h["h"], W, H)
    hx, hy = rect_center(hand_rect)

    print(f"[info] dragging hand[{args.hand}] ({hx},{hy}) -> tile({args.row},{args.col}) center ({tx},{ty}), duration={args.duration}ms")

    if args.dry:
        print("[dry-run] not sending swipe (use without --dry to execute).")
        return

    # do the swipe
    adb_swipe(hx, hy, tx, ty, args.duration)
    print("[ok] swipe sent")

if __name__ == "__main__":
    main()
