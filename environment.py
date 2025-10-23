#!/usr/bin/env python3
"""
environment.py — vision helpers for Merge Tactics
- Loads geometry.json to get normalized ROIs.
- Provides read_mana(img) using OCR and read_cards(img) using ORB.

Assumptions:
- Screenshots are 720x1280 (portrait). All geometry is normalized (0..1).
- Digit templates exist at assets/templates/0.png ... 9.png (prefer white digits on black bg).
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import string
from PIL import Image
from typing import List, Tuple, Union
import json


import cv2
import numpy as np
import pytesseract


GEOM_PATH = Path("geometry.json")
TEMPLATES_DIR = Path("assets/templates")
RUNS_DIR = Path("runs") / "vision"
CARD_REFS_DIR = Path("assets/cards")  
CARDS_INFO_PATH = Path("cards.json")
_CARD_INFO: dict[str, dict] | None = None
TESS_MULTI  = "--psm 10  --oem 1 -c tessedit_char_whitelist=0123456789"

# Utilities
ImgLike = Union[str, Path, np.ndarray]

def _as_bgr(img: ImgLike) -> np.ndarray:
    if isinstance(img, np.ndarray):
        # assume already BGR (as returned by cv2)
        return img
    # else treat as path
    img = cv2.imread(str(img), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {img}")
    return img

def load_card_info(path: Path = CARDS_INFO_PATH) -> dict:
    """
    Load cards.json, which contains cost/type/traits for each unit.
    Caches globally for repeated calls.
    """
    global _CARD_INFO
    if _CARD_INFO is None:
        if not path.exists():
            raise FileNotFoundError(f"cards.json not found at {path}")
        with open(path, "r") as f:
            _CARD_INFO = json.load(f)
    return _CARD_INFO

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

    V_min = 170
    S_max = 250
   
    mask = cv2.inRange(hsv, (0, 0, V_min), (180, S_max, 255)) 

    # Clean mask: close tiny gaps, remove specks
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k, iterations=1)

    res = 255 - mask
    return res

def _has_upgrade_green(roi_bgr: np.ndarray,
                       h_lo: int = 45, h_hi: int = 85,
                       s_lo: int = 120, v_lo: int = 180,
                       min_frac: float = 0.02) -> bool:
    """
    Returns True if a bright green blob is present.
    Defaults tuned for the neon/bright upgrade arrow.
    - min_frac: minimum fraction of green pixels required (2% is a good start).
    """
    if roi_bgr.size == 0:
        return False
    hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (h_lo, s_lo, v_lo), (h_hi, 255, 255))

    # clean small specks
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=1)

    frac = float(np.count_nonzero(mask)) / float(mask.size)
    return frac >= min_frac

def _is_blue_stripe(roi_bgr: np.ndarray,
                    x_frac: float,
                    stripe_frac: float = 0.06,
                    h_lo: int = 95, h_hi: int = 130,  # OpenCV Hue 0..180; 100~120 ≈ deep blue/cyan
                    s_lo: int = 120, v_lo: int = 120,
                    min_blue_frac: float = 0.25) -> bool:
    """
    Returns True if a vertical stripe centered at x_frac contains enough 'blue' pixels.
    stripe_frac: stripe width as fraction of ROI width.
    min_blue_frac: required fraction of blue pixels within the stripe.
    """
    if roi_bgr.size == 0:
        return False

    Ht, Wd = roi_bgr.shape[:2]
    w = max(1, int(Wd * stripe_frac))
    cx = int(Wd * np.clip(x_frac, 0.0, 1.0))
    x0 = max(0, cx - w // 2); x1 = min(Wd, cx + w // 2)
    stripe = roi_bgr[:, x0:x1]

    hsv = cv2.cvtColor(stripe, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (h_lo, s_lo, v_lo), (h_hi, 255, 255))

    # small cleanup
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)

    frac = float(np.count_nonzero(mask)) / float(mask.size)
    return frac >= min_blue_frac
    
# Geometry
def load_geometry(path: Path = GEOM_PATH) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def mana_norm_roi(geom: dict) -> Tuple[float,float,float,float]:
    r = geom["mana"]
    return (float(r["x"]), float(r["y"]), float(r["w"]), float(r["h"]))

def health_abs_rect(geom: dict, W: int, H: int) -> tuple[int,int,int,int]:
    r = geom["health"]
    return _to_abs_rect((float(r["x"]), float(r["y"]), float(r["w"]), float(r["h"])), W, H)

def hand_abs_rects(geom: dict, W: int, H: int, inset_norm: float = 0.04) -> List[Tuple[int,int,int,int]]:
    """
    Build per-hand art rectangles. Uses the full card rect,
    then applies a small normalized inset to stay off the silver/rarity frame.
    """
    rects = []
    for card in geom.get("hand", []):
        cx, cy = float(card["cx"]), float(card["cy"])
        cw, ch = float(card["w"]),  float(card["h"])
        x = (cx - cw/2.0) + inset_norm * cw
        y = (cy - ch/2.0) + inset_norm * ch
        w = cw * (1.0 - 2*inset_norm)
        h = ch * (1.0 - 2*inset_norm)
        rects.append(_to_abs_rect((x, y, w, h), W, H))
    return rects

def hand_abs_upgrade_rects(
        geom: dict, W: int, H: int,
        dx_norm: float = 0.0, dy_norm: float = 0.0, scale: float = 1.0
    ) -> List[Tuple[int,int,int,int]]:
    """
    Returns per-hand absolute rects for the green upgrade arrow, using hand[*].upgrade_arrow_rel.
    The rel rect is expressed in card-relative normalized coords (0..1 of card w/h).
    Optional dx_norm/dy_norm nudge in screen-normalized coords; 'scale' uniformly scales the rect.
    """
    rects = []
    for card in geom.get("hand", []):
        cx, cy = float(card["cx"]), float(card["cy"])
        cw, ch = float(card["w"]),  float(card["h"])
        card_x = cx - cw / 2.0
        card_y = cy - ch / 2.0

        rel = card.get("upgrade_arrow_rel", {"x":0,"y":0,"w":0.2,"h":0.2})
        # anchor in screen-normalized coordinates
        x_norm = card_x + float(rel.get("x", 0.0)) * cw + dx_norm
        y_norm = card_y + float(rel.get("y", 0.0)) * ch + dy_norm
        w_norm = float(rel.get("w", 0.2)) * cw * scale
        h_norm = float(rel.get("h", 0.2)) * ch * scale

        rects.append(_to_abs_rect((x_norm, y_norm, w_norm, h_norm), W, H))
    return rects

def _match_digit(roi: np.ndarray, cfg: str) -> string:
    """
    Returns (digits_only, avg_conf). Conf is mean over symbols with non-negative conf,
    as reported by pytesseract.image_to_data.
    """
    if roi.ndim == 3:
        roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    data = pytesseract.image_to_data(roi, config=cfg, output_type=pytesseract.Output.DICT)

    digits: List[str] = []
    confs: List[float] = []
    n = len(data.get("text", []))

    for i in range(n):
        raw = (data["text"][i] or "").strip()
        if not raw:
            continue
        ds = "".join(ch for ch in raw if ch.isdigit())
        if not ds:
            continue
        digits.append(ds)
        try:
            c = float(data.get("conf", ["-1"])[i])
            if c >= 0:
                confs.append(c)
        except Exception:
            pass

    text = "".join(digits)
    avg_conf = float(np.mean(confs)) if confs else -1.0
    return text, avg_conf

# Card Recognizer Helpers

def _cr_imread_bgr(path: Path) -> np.ndarray:
    buf = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(path)
    if img.ndim == 3 and img.shape[2] == 4:  # composite alpha on black
        b, g, r, a = cv2.split(img)
        a = a.astype(np.float32) / 255.0
        b = (b * a).astype(np.uint8); g = (g * a).astype(np.uint8); r = (r * a).astype(np.uint8)
        img = cv2.merge([b, g, r])
    elif img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img

def _cr_preprocess_art(bgr: np.ndarray, inset: float) -> np.ndarray:
    h, w = bgr.shape[:2]
    dx, dy = int(w * inset), int(h * inset)
    x0, y0, x1, y1 = dx, dy, w - dx, h - dy
    crop = bgr[y0:y1, x0:x1] if (x1 > x0 and y1 > y0) else bgr
    crop = cv2.resize(crop, (192, 256), interpolation=cv2.INTER_AREA)
    blur = cv2.GaussianBlur(crop, (0, 0), 1.0)
    sharp = cv2.addWeighted(crop, 1.5, blur, -0.5, 0)
    return sharp

def _cr_phash(bgr: np.ndarray) -> np.ndarray:
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    g = cv2.resize(g, (32, 32), interpolation=cv2.INTER_AREA)
    dct = cv2.dct(g.astype(np.float32))
    d = dct[:8, :8]
    med = np.median(d[1:, 1:])
    bits = (d >= med).astype(np.uint8).flatten()
    return np.packbits(bits)  # uint8[8]

def _cr_hamming_batch(hash8: np.ndarray, ref_hashes8: np.ndarray) -> np.ndarray:
    x = np.bitwise_xor(ref_hashes8, hash8[None, :])
    lut = np.unpackbits(np.arange(256, dtype=np.uint8)[:, None], axis=1).sum(1)
    return lut[x].sum(axis=1)

# Card Recognizer
@dataclass
class CardMatch:
    label: str | None
    conf: float                  
    method: str
    details: dict
    upgradable: bool = False


    # Extra metadata from cards.json
    cost: int | None = None
    type: str | None = None
    trait1: str | None = None
    trait2: str | None = None

class _CardRecognizer:
    """pHash shortlist + ORB re-ranking"""
    def __init__(self, ref_dir: Path = CARD_REFS_DIR, inset: float = 0.1, orb_nfeatures: int = 600):
        self.ref_dir = Path(ref_dir)
        self.inset = inset
        self.orb = cv2.ORB_create(nfeatures=orb_nfeatures, scaleFactor=1.2, edgeThreshold=15, patchSize=31)
        self.bfm = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

        self.labels: list[str] = []
        self.ref_phashes: np.ndarray | None = None  # [N, 8] uint8 (64-bit hash)
        self.ref_kp: list[list[cv2.KeyPoint]] = []
        self.ref_desc: list[np.ndarray | None] = []

        self._load_refs()

    def _load_refs(self):
        if not self.ref_dir.exists():
            raise RuntimeError(f"Card refs folder not found: {self.ref_dir}")
        exts = {".webp", ".png", ".jpg", ".jpeg"}
        paths = sorted([p for p in self.ref_dir.iterdir() if p.suffix.lower() in exts])
        if not paths:
            raise RuntimeError(f"No reference images in {self.ref_dir}")

        labels, hashes, kps, descs = [], [], [], []
        for p in paths:
            lab = p.stem
            img = _cr_imread_bgr(p)
            art = _cr_preprocess_art(img, self.inset)
            h = _cr_phash(art)
            kp, d = self.orb.detectAndCompute(art, None)
            labels.append(lab); hashes.append(h); kps.append(kp or []); descs.append(d)
        self.labels = labels
        self.ref_phashes = np.stack(hashes, axis=0).astype(np.uint8)
        self.ref_kp = kps
        self.ref_desc = descs

    def predict(self, roi_bgr: np.ndarray, topk: int = 6) -> CardMatch:
        q = _cr_preprocess_art(roi_bgr, self.inset)
        q_hash = _cr_phash(q)
        q_kp, q_desc = self.orb.detectAndCompute(q, None)

        # shortlist by pHash (Hamming)
        dists = _cr_hamming_batch(q_hash, self.ref_phashes)
        idx_sorted = np.argsort(dists)[:max(1, min(topk, len(self.labels)))]

        best_idx, best_score, best_good = -1, -1.0, 0
        for idx in idx_sorted:
            ref_d = self.ref_desc[idx]
            if q_desc is None or ref_d is None or len(ref_d) == 0:
                continue
            matches = self.bfm.knnMatch(q_desc, ref_d, k=2)
            good = 0
            for m, n in matches:
                if m.distance < 0.75 * n.distance:  # Lowe ratio
                    good += 1
            denom = max(1.0, np.sqrt((len(q_kp or []) + len(self.ref_kp[idx] or [])) / 2.0))
            score = good / denom
            if score > best_score:
                best_idx, best_score, best_good = idx, score, good

        conf = 1.0 / (1.0 + np.exp(-4.0 * (best_score - 0.7)))  # squashed, center ~0.7

        if best_idx < 0:
            return CardMatch(label=None, conf=0.0, method="phash+orb",
                             details={"orb_score": 0.0, "good": 0, "phash_best": float(dists.min())})
        return CardMatch(
            label=self.labels[best_idx],
            conf=float(conf),
            method="phash+orb",
            details={"orb_score": float(best_score), "good": int(best_good), "phash_dist": float(dists[best_idx])},
        )

# Public APIs

_CARD_MODEL: _CardRecognizer | None = None

def init_card_model(ref_dir: Path | str = CARD_REFS_DIR) -> None:
    """Call once at startup (or lazy-init will happen on first use)."""
    global _CARD_MODEL
    _CARD_MODEL = _CardRecognizer(Path(ref_dir))

def read_cards(img: ImgLike) -> List[CardMatch]:
    """
    Recognize each hand card
    Returns list[CardMatch].
    """
    global _CARD_MODEL
    if _CARD_MODEL is None:
        init_card_model(CARD_REFS_DIR)

    frame_bgr = _as_bgr(img)
    H, W = frame_bgr.shape[:2]
    geom = load_geometry()
    card_rects = hand_abs_rects(geom, W, H)
    upg_rects = hand_abs_upgrade_rects(geom, W, H)

    card_info = load_card_info()

    results: list[CardMatch] = []
    for i in range(len(card_rects)):
        roi = _crop(frame_bgr, card_rects[i])
        match = _CARD_MODEL.predict(roi)

        if match.label and match.label in card_info:
            info = card_info[match.label]
            match.cost = info.get("cost")
            match.type = info.get("type")
            match.trait1 = info.get("trait1")
            match.trait2 = info.get("trait2")
   
        roi_upg = _crop(frame_bgr, upg_rects[i])
        match.upgradable = _has_upgrade_green(roi_upg)
        results.append(match)
    return results


def read_mana(img: ImgLike) -> Tuple[str, float]:
    """
    Read the mana pool. Returns (text, avg_conf).
    - text may contain multiple digits (e.g., '10')
    - avg_conf in 0..100, or -1 if nothing recognized
    """
    frame_bgr = _as_bgr(img)
    H, W = frame_bgr.shape[:2]

    geom = load_geometry()
    rect = _to_abs_rect(mana_norm_roi(geom), W, H)
    roi = _crop(frame_bgr, rect)
    roi = _clean_roi(roi)

    text, conf = _match_digit(roi, TESS_MULTI)
    return text, conf

def read_health(img: ImgLike) -> int:
    """
    Quantized health from the blue bar:
      - Probe stripes at 25%, 50%, 75% of the health ROI width.
      - If 75% stripe is blue ⇒ ≥75% (then check 100% by looking near the far-right edge).
      - If 50% stripe is blue ⇒ ≥50%.
      - If 25% stripe is blue ⇒ ≥25%.
      - Else 0%.
    Returns: int representing percent
    """
    frame_bgr = _as_bgr(img)
    H, W = frame_bgr.shape[:2]
    geom = load_geometry()
    rx, ry, rw, rh = health_abs_rect(geom, W, H)

    roi = _crop(frame_bgr, (rx, ry, rw, rh))

    p10  = _is_blue_stripe(roi, 0.10)
    p20  = _is_blue_stripe(roi, 0.20)
    p30  = _is_blue_stripe(roi, 0.30)
    p40  = _is_blue_stripe(roi, 0.40)
    p50  = _is_blue_stripe(roi, 0.50)
    p60  = _is_blue_stripe(roi, 0.60)
    p70  = _is_blue_stripe(roi, 0.70)
    p80  = _is_blue_stripe(roi, 0.80)
    p90  = _is_blue_stripe(roi, 0.90)
    p100 = _is_blue_stripe(roi, 0.97, stripe_frac=0.04)

    percent = 0
    for level, probe in ([
        (100, p100), (90, p90), (80, p80), (70, p70),
        (60, p60), (50, p50), (40, p40), (30, p30),
        (20, p20), (10, p10)
    ]):
        if probe:
            percent = level
            break

    return percent


# CLI (quick manual test)
if __name__ == "__main__":
    import argparse, sys

    ap = argparse.ArgumentParser(description="Read mana and hand costs via template matching using geometry.json")
    ap.add_argument("image", type=str, help="Path to screenshot image")
    args = ap.parse_args()

    mana = read_mana(args.image)
    print(f"Mana: {mana[0]} ({mana[1]})")

    health = read_health(args.image)
    print(f"Health: {health}")

 