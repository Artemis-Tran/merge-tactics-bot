#!/usr/bin/env python3
"""
environment.py — vision helpers for Merge Tactics
- Loads geometry.json to get normalized ROIs.
- Provides read_mana(image_path) and read_hand_costs(image_path) using template matching.

Assumptions:
- Screenshots are 720x1280 (portrait). All geometry is normalized (0..1).
- Digit templates exist at assets/templates/0.png ... 9.png (prefer white digits on black bg).
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import string
from PIL import Image
from typing import List, Tuple
import json


import cv2
import numpy as np
import pytesseract


GEOM_PATH = Path("geometry.json")
TEMPLATES_DIR = Path("assets/templates")
RUNS_DIR = Path("runs") / "vision"


# Utilities
def _to_abs_rect(norm_rect: Tuple[float,float,float,float], W: int, H: int) -> Tuple[int,int,int,int]:
    x,y,w,h = norm_rect
    return (int(x*W), int(y*H), int(w*W), int(h*H))


def _crop(img: np.ndarray, rect: Tuple[int,int,int,int]) -> np.ndarray:
    x,y,w,h = rect
    return img[max(0,y):y+h, max(0,x):x+w].copy()


def _clean_roi(img: np.ndarray) -> np.ndarray:
    img = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

    # Segment the light digit by color: high value (brightness), low saturation
    hsv  = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    H,S,V = cv2.split(hsv)

    V_min = 230 
    S_max = 40   
   
    mask = cv2.inRange(hsv, (0, 0, V_min), (180, S_max, 255)) 

    # Clean mask: close tiny gaps, remove specks
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k, iterations=1)

    res = 255 - mask
    return res

# Geometry
def load_geometry(path: Path = GEOM_PATH) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def mana_norm_roi(geom: dict) -> Tuple[float,float,float,float]:
    r = geom["mana_roi"]
    return (float(r["x"]), float(r["y"]), float(r["w"]), float(r["h"]))


def _mana_aspect_and_height(geom: dict) -> Tuple[float, float]:
    mr = geom["mana_roi"]
    aspect = float(mr["w"]) / float(mr["h"])  # W/H of mana pool
    h_norm = float(mr["h"])                   # normalized height of mana pool
    return aspect, h_norm


def hand_abs_cost_rects(
    geom: dict, W: int, H: int, height_scale: float = 0.75, dx_norm: float = -0.003, dy_norm: float = 0.005
) -> List[Tuple[int,int,int,int]]:
    """
    Build hand cost rects:
      - Anchor = card top-left + (cost_roi_rel.x, cost_roi_rel.y) * card size
      - Size   = mana aspect ratio; height = mana_roi.h * height_scale (both normalized to screen)
    Optional dx_norm/dy_norm let you nudge the box in screen-normalized coords if needed.
    """
    aspect, mana_h = _mana_aspect_and_height(geom)
    h_norm = mana_h * height_scale
    w_norm = aspect * h_norm

    rects = []
    for card in geom.get("hand", []):
        cx, cy = float(card["cx"]), float(card["cy"])
        cw, ch = float(card["w"]),  float(card["h"])
        card_x = cx - cw / 2.0
        card_y = cy - ch / 2.0

        rel = card["cost_roi_rel"]
        anchor_x = card_x + float(rel["x"]) * cw + dx_norm
        anchor_y = card_y + float(rel["y"]) * ch + dy_norm

        cost_norm = (anchor_x, anchor_y, w_norm, h_norm)
        x, y, w, h = _to_abs_rect(cost_norm, W, H)
        rects.append((x, y, w, h))
    return rects

def _match_digit(roi: np.ndarray) -> string:
    """Given a **single-digit** ROI (already cropped), uses OCR to find digit."""
    pilImg = Image.fromarray(roi)
    text = pytesseract.image_to_string(pilImg, config='--psm 7 -c tessedit_char_whitelist=0123456789')
    return text.strip()


# Public API
def read_mana(image_path: Path | str) -> np.ndarray:
    """
    Reads the *mana pool* digit from the ROI defined in geometry.json using template matching.
    Returns total mana
    """
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Could not open image: {image_path}")
    H, W = img.shape[:2]

    geom = load_geometry()
    roi_norm = mana_norm_roi(geom)
    abs_rect = _to_abs_rect(roi_norm, W, H)
    roi = _crop(img, abs_rect)
    roi = _clean_roi(roi)
   
    res = _match_digit(roi)
    return res


def read_hand_costs(image_path: Path | str):
    """
    Reads the visible *hand* card costs using the cost ROI relative to each card rect from geometry.json.
    Returns a list of the mana costs
    """
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Could not open image: {image_path}")
    H, W = img.shape[:2]

    geom = load_geometry()
    rects = hand_abs_cost_rects(geom, W, H, height_scale=0.5)

    results = []
    for _, rect in enumerate(rects):
        roi = _crop(img, rect)
        roi = _clean_roi(roi)
        res = _match_digit(roi)
        results.append(res)

    return results


# CLI (quick manual test)
if __name__ == "__main__":
    import argparse, sys

    ap = argparse.ArgumentParser(description="Read mana and hand costs via template matching using geometry.json")
    ap.add_argument("image", type=str, help="Path to screenshot image")
    ap.add_argument("--what", type=str, default="mana", choices=["mana", "hand", "both"],
                    help="Which read to perform")
    args = ap.parse_args()

    if args.what in ("mana", "both"):
        res = read_mana(args.image)
        print(f"Mana: {res}")
    if args.what in ("hand", "both"):
        res = read_hand_costs(args.image)
        for d in res:
            print(f"Hand: {d}")
 