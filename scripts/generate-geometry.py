#!/usr/bin/env python3
"""
generate_geometry.py

Creates geometry.json and annotated_geometry_overlay.png for Merge Tactics.
Saves geometry.json in the current directory and overlay PNG in ./screenshots.

Adds a cost ROI for each hand card (top-left corner of the card) so you can OCR the mana cost later.
"""

import argparse
import json
from pathlib import Path
import cv2
import numpy as np


def to_norm_rect(x, y, w, h, W, H):
    return {"x": x / W, "y": y / H, "w": w / W, "h": h / H}


def to_abs_rect(nr, W, H):
    return (
        int(nr["x"] * W),
        int(nr["y"] * H),
        int(nr["w"] * W),
        int(nr["h"] * H),
    )


def draw_labeled_rect(img, rect, color, label=None, thickness=2):
    x, y, w, h = rect
    cv2.rectangle(img, (x, y), (x + w, y + h), color, thickness)
    if label:
        cv2.putText(
            img, label, (x, max(18, y - 6)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA
        )


def find_board_rect(bgr):
    H, W = bgr.shape[:2]
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    lower_green = np.array([40, 40, 40])
    upper_green = np.array([85, 255, 255])
    mask = cv2.inRange(hsv, lower_green, upper_green)
    kernel = np.ones((11, 11), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        x, y, w, h = int(0.18 * W), int(0.25 * H), int(0.64 * W), int(0.55 * H)
    else:
        c = max(cnts, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)
    x_in = x + int(0.03 * w)
    y_in = y + int(0.04 * h)
    w_in = w - int(0.06 * w)
    h_in = h - int(0.08 * h)
    return (x_in, y_in, w_in, h_in), hsv


def split_playable_half(board_rect):
    x, y, w, h = board_rect
    py = y + h // 2
    ph = h - (py - y)
    playable = (x, py, w, ph)
    return board_rect, playable


def hex_centers_from_rect(rect_px, rows, cols, W, H):
    bx, by, bw, bh = rect_px
    centers = []

    col_spacing_factor = 0.85   # smaller -> less horizontal space
    row_spacing_factor = 0.9  # smaller -> less vertical space

    dx = (bw / cols) * col_spacing_factor
    dy = (bh / rows) * row_spacing_factor
    for r in range(rows):
        y = by + (r + 0.5) * dy
        x_shift = dx / 2 if (r % 2 == 0) else 0.0
        for c in range(cols):
            x = bx + x_shift + (c + 0.725) * dx
            centers.append((x / W, y / H))
    return centers, dx, dy


def rel_child_abs_norm(parent_center_norm, parent_size_norm, child_rel):
    """
    Convert a child ROI given relative to a center-sized parent box into absolute normalized rect.
    parent_center_norm = (cx, cy) normalized center of parent
    parent_size_norm   = (w, h) normalized size of parent
    child_rel          = {"x","y","w","h"} relative (0..1) inside parent box (origin at parent's top-left)
    Returns absolute normalized {"x","y","w","h"}.
    """
    pcx, pcy = parent_center_norm
    pw, ph = parent_size_norm
    # parent top-left in norm
    p_x = pcx - pw / 2.0
    p_y = pcy - ph / 2.0
    ax = p_x + child_rel["x"] * pw
    ay = p_y + child_rel["y"] * ph + 0.005
    aw = child_rel["w"] * pw
    ah = child_rel["h"] * ph - 0.016
    return {"x": ax, "y": ay, "w": aw, "h": ah}


def detect_mana_roi(hsv):
    H, W = hsv.shape[:2]
    search = hsv[int(0.82 * H):H, int(0.65 * W):W]
    lower_orange = np.array([8, 120, 120])
    upper_orange = np.array([25, 255, 255])
    mask = cv2.inRange(search, lower_orange, upper_orange)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if cnts:
        c = max(cnts, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)
        return {
            "x": (int(0.65 * W) + x) / W,
            "y": (int(0.825 * H) + y) / H,
            "w": w / W,
            "h": h / H - 0.01,
        }
    return {"x": 0.86, "y": 0.90, "w": 0.10, "h": 0.07}


def compute_bench_slots(playable_rect_px, hand_boxes_norm, W, H, n_slots=5, bench_width_factor=0.84):
    """
    Build n_slots contiguous bench rectangles between the bottom of the playable board and the top of the hand.
    The bench is left-aligned to the playable board's left edge but only uses a fraction of its width.
      - bench_width_factor: fraction of playable width to occupy (0<factor<=1), e.g. 0.86
      - left-most slot aligns to playable left; remaining width becomes a right margin.
    """
    bx, by, bw, bh = playable_rect_px
    board_bottom = by + bh

    if not hand_boxes_norm:
        return []

    # Hand top (from first hand box)
    hb = hand_boxes_norm[0]
    hand_top_px = int(hb["cy"] * H - (hb["h"] * H) / 2)

    # Vertical band between board bottom and hand top (with a small pad)
    pad = int(0.02 * H)
    y0 = board_bottom + pad
    y1 = hand_top_px - pad
    if y1 <= y0:
        band_h = int(0.08 * H)
        y0 = board_bottom + pad
        y1 = y0 + band_h
    band_h = y1 - y0 - 20

    # Bench width is a fraction of the playable width, anchored to the left
    bench_w = int(bw * bench_width_factor)
    bench_x0 = bx  # left-aligned
    # right margin exists implicitly: (bx + bw) - (bench_x0 + bench_w)

    # Make contiguous slot widths that sum exactly to bench_w
    base_w = bench_w // n_slots
    remainder = bench_w % n_slots  # distribute leftover 1px across first 'remainder' slots

    slots = []
    x_cursor = bench_x0
    for i in range(n_slots):
        w_i = base_w + (1 if i < remainder else 0)
        slots.append(to_norm_rect(x_cursor, y0, w_i, band_h, W, H))
        x_cursor += w_i

    return slots
    
    


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img", type=Path, default=Path("assets/screenshots/beginning-board.png"))
    parser.add_argument("--rows", type=int, default=4)
    parser.add_argument("--cols", type=int, default=5)
    parser.add_argument("--out_dir", type=Path, default=Path("../screenshots"))
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    if not args.img.exists():
        raise FileNotFoundError(f"Image not found: {args.img}")

    bgr = cv2.imread(str(args.img))
    if bgr is None:
        raise RuntimeError(f"Could not load image: {args.img}")

    H, W = bgr.shape[:2]

    board_rect, hsv = find_board_rect(bgr)
    full_rect = board_rect
    full_norm = to_norm_rect(*full_rect, W, H)

    _, playable_rect = split_playable_half(full_rect)
    px, py, pw, ph = playable_rect
    playable_norm = to_norm_rect(px, py, pw, ph, W, H)

    centers, dx, dy = hex_centers_from_rect(playable_rect, args.rows, args.cols, W, H)
    tile_w_norm = (0.6 * dx) / W
    tile_h_norm = (0.6 * dy) / H
    tiles = [{"cx": float(cx), "cy": float(cy), "w": float(tile_w_norm), "h": float(tile_h_norm)}
             for (cx, cy) in centers]

    # 3 hardcoded hand slots
    hand_centers_x = [0.26, 0.42, 0.58]
    hand_y = 0.92
    hand_w, hand_h = 0.15, 0.12

    # cost ROI relative to each hand box (top-left corner of the card)
    # tweak these if your card design differs:
    # x,y,w,h are relative to the hand card rectangle (0..1, origin at hand box top-left)
    cost_roi_rel = {"x": 0.05, "y": 0.05, "w": 0.22, "h": 0.30}

    hand = []
    for xc in hand_centers_x:
        hand_entry = {
            "cx": float(xc),
            "cy": float(hand_y),
            "w": float(hand_w),
            "h": float(hand_h),
            "cost_roi_rel": cost_roi_rel
        }
        hand.append(hand_entry)

    mana_roi = detect_mana_roi(hsv)

    bench_slots = compute_bench_slots(playable_rect, hand, W, H, n_slots=5)

    geometry = {
        "image_basis": str(args.img.name),
        "resolution_px": [W, H],
        "board_rect_full": full_norm,
        "board_rect_playable": playable_norm,
        "board": {"rows": args.rows, "cols": args.cols, "tiles": tiles},
        "bench": {"slots": bench_slots},
        "hand": hand,
        "mana_roi": mana_roi,
    }

    out_json = Path("geometry.json")
    with open(out_json, "w") as f:
        json.dump(geometry, f, indent=2)

    overlay = bgr.copy()

    # draw full and playable rects
    bx, by, bw, bh = full_rect
    draw_labeled_rect(overlay, (bx, by, bw, bh), (0, 255, 0), "BOARD full")
    draw_labeled_rect(overlay, (px, py, pw, ph), (0, 200, 255), "PLAYABLE")

    # draw tiles
    for i, t in enumerate(tiles):
        cx = int(t["cx"] * W)
        cy = int(t["cy"] * H)
        tw = int(t["w"] * W)
        th = int(t["h"] * H)
        cv2.rectangle(overlay, (cx - tw // 2, cy - th // 2),
                      (cx + tw // 2, cy + th // 2), (255, 0, 0), 2)
        cv2.circle(overlay, (cx, cy), 5, (255, 0, 0), -1)
        cv2.putText(overlay, f"{i}", (cx - 8, cy - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
    # draw bench slots
    for i, s in enumerate(bench_slots):
        sx, sy, sw, sh = to_abs_rect(s, W, H)
        draw_labeled_rect(overlay, (sx, sy, sw, sh), (255, 0, 255), f"B{i+1}")

    # draw hand boxes and cost sub-ROIs
    for i, hbox in enumerate(hand):
        hx = int(hbox["cx"] * W)
        hy = int(hbox["cy"] * H)
        hw = int(hbox["w"] * W)
        hh = int(hbox["h"] * H)
        draw_labeled_rect(overlay, (hx - hw // 2, hy - hh // 2, hw, hh),
                          (0, 165, 255), f"H{i+1}")

        # compute absolute normalized cost ROI and draw it
        parent_center_norm = (hbox["cx"], hbox["cy"])
        parent_size_norm = (hbox["w"], hbox["h"])
        cost_abs_norm = rel_child_abs_norm(parent_center_norm, parent_size_norm, hbox["cost_roi_rel"])
        cax, cay, caw, cah = to_abs_rect(cost_abs_norm, W, H)
        draw_labeled_rect(overlay, (cax, cay, caw, cah), (0, 140, 255), f"C{i+1}")

    # draw mana and sell
    mx, my, mw_, mh_ = to_abs_rect(mana_roi, W, H)
    draw_labeled_rect(overlay, (mx, my, mw_, mh_), (0, 255, 255), "MANA")

    out_dir = args.out_dir
    out_overlay = out_dir / "annotated_geometry_overlay.png"
    ok = cv2.imwrite(str(out_overlay), overlay)
    assert ok, f"Failed to save overlay to {out_overlay}"

    print(f"Saved geometry.json in: {out_json.resolve()}")
    print(f"Saved overlay PNG in:  {out_overlay.resolve()}")


if __name__ == "__main__":
    main()
