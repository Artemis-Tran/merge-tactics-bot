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
import io
from pathlib import Path
import re
import string
from PIL import Image
from typing import Dict, List, Optional, Tuple, Union
import json
import time
import os
from dotenv import load_dotenv

import cv2
import numpy as np
import pytesseract
from concurrent.futures import ThreadPoolExecutor
from inference_sdk import InferenceHTTPClient


GEOM_PATH = Path("geometry.json")
END_GEOM_PATH = Path("end-screen-geometry.json")
HOME_GEOM_PATH = Path("home-screen-geometry.json")
TEMPLATES_DIR = Path("assets/templates")
RUNS_DIR = Path("runs") / "vision"
CARD_REFS_DIR = Path("assets/cards")  
CARDS_INFO_PATH = Path("cards.json")
_CARD_INFO: dict[str, dict] | None = None
_TMPL_CACHE: Optional[Dict[int, np.ndarray]] = None
TESS_MULTI  = "--psm 10  --oem 1 -c tessedit_char_whitelist=0123456789"

# Roboflow class name to game label mapping
CLASS_TO_LABEL = {
    "archer": "Archers",
    "archer-queen": "ArcherQueen",
    "baby-dragon": "BabyDragon",
    "bandit": "Bandit",
    "barbarian": "Barbarians",
    "dart-goblin": "DartGoblin",
    "electro-giant": "ElectroGiant",
    "electro-wizard": "ElectroWizard",
    "executioner": "Executioner",
    "giant-skeleton": "GiantSkeleton",
    "goblin-machine": "GoblinMachine",
    "golden-knight": "GoldenKnight",
    "knight": "Knight",
    "mega-knight": "MegaKnight",
    "musketeer": "Musketeer",
    "pekka": "PEKKA",
    "prince": "Prince",
    "princess": "Princess",
    "royal-ghost": "RoyalGhost",
    "skeleton-dragon": "SkeletonDragons",
    "skeleton-king": "SkeletonKing",
    "spear-goblin": "SpearGoblins",
    "stab-goblin": "Goblins",
    "valkyrie": "Valkyrie",
    "witch": "Witch",
    "wizard": "Wizard",
}

# Helpers
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

# Cleaning for OCR
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

def load_digit_templates(dirpath: Path = TEMPLATES_DIR) -> dict[int, np.ndarray]:
    global _TMPL_CACHE
    if _TMPL_CACHE is not None:
        return _TMPL_CACHE
    tmpls: dict[int, np.ndarray] = {}
    for d in range(10):
        p = dirpath / f"{d}.png"
        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Could not read digit template: {p}")
        tmpls[d] = img
    _TMPL_CACHE = tmpls
    return tmpls


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

def get_abs_rect(geom: dict, key: str, W: int, H: int) -> Tuple[int,int,int,int]:
    """Return absolute rect (x, y, w, h) in pixels for a top-level geometry key."""
    if key not in geom:
        raise KeyError(f"No ROI named '{key}' in geometry.json")
    r = geom[key]
    return _to_abs_rect((float(r["x"]), float(r["y"]), float(r["w"]), float(r["h"])), W, H)

def hand_abs_rects(geom: dict, W: int, H: int, inset_norm: float = 0.05) -> List[Tuple[int,int,int,int]]:
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

def coords_to_tile_index(x: float, y: float, geometry: dict) -> Optional[int]:
    """Maps pixel coordinates to a board tile index."""
    img_w, img_h = geometry["resolution_px"]
    for i, tile in enumerate(geometry["board"]["tiles"]):
        # Convert relative tile geometry to absolute pixel coordinates
        cx, cy, w, h = tile["cx"], tile["cy"], tile["w"], tile["h"]
        x1 = (cx - w / 2) * img_w
        y1 = (cy - h / 2) * img_h
        x2 = (cx + w / 2) * img_w
        y2 = (cy + h / 2) * img_h
        
        if x1 <= x <= x2 and y1 <= y <= y2:
            return i
    return None

def segment_digits(bin_img: np.ndarray, tmpls: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Split a binarized image (white bg=255, black digits=0) into two digits
    and pad each to the template size using the template's foreground bbox.
    """
    def tight_bbox_black_on_white(img: np.ndarray):
        g = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        fg = (g < 128).astype(np.uint8)
        ys, xs = np.where(fg > 0)
        if xs.size == 0 or ys.size == 0:
            return 0, 1, 0, 1
        return ys.min(), ys.max()+1, xs.min(), xs.max()+1

    def pad_to_match_template_bbox(digit_img: np.ndarray, template_img: np.ndarray):

        Ht, Wt = template_img.shape[:2]
        y0t, y1t, x0t, x1t = tight_bbox_black_on_white(template_img)
        y0, y1, x0, x1 = tight_bbox_black_on_white(digit_img)
        crop = digit_img[y0:y1, x0:x1]

        h, w = crop.shape
        target_h, target_w = y1t - y0t, x1t - x0t
        scale = min(target_h / max(1, h), target_w / max(1, w))
        nh, nw = max(1, int(round(h * scale))), max(1, int(round(w * scale)))
        resized = cv2.resize(crop, (nw, nh), interpolation=cv2.INTER_AREA)

        canvas = np.full((Ht, Wt), 255, np.uint8)
        y_off = y0t + (target_h - nh)//2
        x_off = x0t + (target_w - nw)//2
        canvas[y_off:y_off+nh, x_off:x_off+nw] = resized
        return canvas

    g = bin_img if bin_img.ndim == 2 else cv2.cvtColor(bin_img, cv2.COLOR_BGR2GRAY)
    fg = (g < 128).astype(np.uint8)

    proj = fg.sum(axis=0).astype(np.float32)
    proj_s = np.convolve(proj, np.ones(15, np.float32)/15, mode="same")
    W = proj_s.shape[0]
    L, R = int(W*0.35), int(W*0.65)
    split = L + int(np.argmin(proj_s[L:R]))

    def crop_side(mask: np.ndarray):
        ys, xs = np.where(mask > 0)
        if xs.size == 0 or ys.size == 0:
            return np.full((1, 1), 255, np.uint8)
        x0, x1 = xs.min(), xs.max()+1
        y0, y1 = ys.min(), ys.max()+1
        return ((1 - mask[y0:y1, x0:x1]) * 255).astype(np.uint8)

    left = crop_side(fg[:, :split])
    right = crop_side(fg[:, split:])
    tmpl_any = next(iter(tmpls.values()))
    return (
        pad_to_match_template_bbox(left, tmpl_any),
        pad_to_match_template_bbox(right, tmpl_any),
    )

def _match_digit(roi: np.ndarray, cfg: str) -> Tuple[str, float]:
    """
    Returns (digits_only, avg_conf). Conf is mean over symbols with non-negative conf,
    as reported by pytesseract.image_to_data.
    """

    H, W = roi.shape[:2]
    tmpls = load_digit_templates()
    best_digit, best_score = -1, -1.0
    for digit, template in tmpls.items():
        score = float(cv2.matchTemplate(roi, template, cv2.TM_CCOEFF_NORMED)[0][0])
        if score > best_score:
            best_digit, best_score = digit, score

    if best_score >= 0.65:
        return str(best_digit), round(best_score, 2) * 100.0
    
    # Must be double digit
    left_digit, right_digit = segment_digits(roi, tmpls)

    best_left_digit, best_left_score = -1, -1.0
    best_right_digit, best_right_score = -1, -1.0
    for digit, template in tmpls.items():
        left_score = float(cv2.matchTemplate(left_digit, template, cv2.TM_CCOEFF_NORMED)[0][0])
        right_score = float(cv2.matchTemplate(right_digit, template, cv2.TM_CCOEFF_NORMED)[0][0])
        if left_score > best_left_score:
            best_left_digit, best_left_score = digit, left_score
        if right_score > best_right_score:
            best_right_digit, best_right_score = digit, right_score

    best_digit = str(best_left_digit) + str(best_right_digit)
    best_score = (best_left_score + best_right_score) / 2.0
        
    return str(best_digit), round(best_score, 2) * 100.0

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
    idx: int | None = None
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

    def predict(self, roi_bgr: np.ndarray, topk: int = 4) -> CardMatch:
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
                if m.distance < 0.75 * n.distance: 
                    good += 1
            denom = max(1.0, np.sqrt((len(q_kp or []) + len(self.ref_kp[idx] or [])) / 2.0))
            score = good / denom
            if score > best_score:
                best_idx, best_score, best_good = idx, score, good

        conf = 1.0 / (1.0 + np.exp(-4.0 * (best_score - 0.7)))  # squashed, center ~0.7

        if best_idx < 0.5:
            return CardMatch(label=None, conf=0.0, method="phash+orb",
                             details={"orb_score": 0.0, "good": 0, "phash_best": float(dists.min())})
        return CardMatch(
            label=self.labels[best_idx],
            conf=float(conf),
            method="phash+orb",
            details={"orb_score": float(best_score), "good": int(best_good), "phash_dist": float(dists[best_idx])},
        )

# Public APIs

@dataclass
class GameState:
    mana: int
    mana_conf: float
    health: int                       # {0,10,20,...,100}
    round: Optional[int]
    phase: Optional[str]                  # 'deploy' or 'battle'
    cards: List[CardMatch]                # matches for cards in hand
    game_over: bool
    timer: Optional[int]

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
    def process_card(i: int) -> CardMatch:
        roi = _crop(frame_bgr, card_rects[i])
        match = _CARD_MODEL.predict(roi) 

        # enrich metadata
        if match.label and match.label in card_info:
            info = card_info[match.label]
            match.idx = i
            match.cost = info.get("cost")
            match.type = info.get("type")
            match.trait1 = info.get("trait1")
            match.trait2 = info.get("trait2")

        # upgrade arrow
        roi_upg = _crop(frame_bgr, upg_rects[i])
        match.upgradable = _has_upgrade_green(roi_upg)
        return match
    with ThreadPoolExecutor(max_workers=3) as ex:
        results = list(ex.map(process_card, range(len(card_rects))))
    return results


def read_mana(img: ImgLike) -> Tuple[int, float]:
    """
    Read the mana pool. Returns (text, avg_conf).
    - text may contain multiple digits (e.g., '10')
    - avg_conf in 0..100, or -1 if nothing recognized
    """
    frame_bgr = _as_bgr(img)
    H, W = frame_bgr.shape[:2]

    geom = load_geometry()
    rect = get_abs_rect(geom, "mana", W, H)
    roi = _crop(frame_bgr, rect)
    roi = _clean_roi(roi)

    text, conf = _match_digit(roi, TESS_MULTI)

    if conf < 20 or not text or not str(text).isdigit():
        return 0, -1.0

    val = max(0, min(20, int(text)))
    return val, conf

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
    rx, ry, rw, rh = get_abs_rect(geom, "health", W, H)

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

def read_timer_percent(img: ImgLike) -> int:
    """
    Estimate remaining time (0..100) from the blue timer bar.
    Logic: scan vertical stripes from right->left; the rightmost blue stripe
    gives the remaining fraction.
    """
    frame_bgr = _as_bgr(img)
    H, W = frame_bgr.shape[:2]
    geom = load_geometry()

    # geometry.json already has a "timer" ROI (normalized full-width thin bar)
    rx, ry, rw, rh = get_abs_rect(geom, "timer", W, H)
    roi = _crop(frame_bgr, (rx, ry, rw, rh))

    # Search from right (1.0) to left (0.0) in small steps
    rightmost_blue_x = None
    step = 0.02
    x = 0.98
    while x >= 0.0:
        if _is_blue_stripe(roi, x_frac=x, stripe_frac=0.06,
                           h_lo=95, h_hi=130, s_lo=120, v_lo=120,  # same blue band as health
                           min_blue_frac=0.25):
            rightmost_blue_x = x
            break
        x -= step

    if rightmost_blue_x is None:
        return 0
    pct = int(round(rightmost_blue_x * 100))
    if rightmost_blue_x >= 0.97:
        pct = 100
    return max(0, min(100, pct))

def read_round_phase(img: ImgLike) -> Tuple[int, str]:
    frame_bgr = _as_bgr(img)
    H, W = frame_bgr.shape[:2]
    geom = load_geometry()
    rx, ry, rw, rh = get_abs_rect(geom, "round_phase", W, H)

    roi = _crop(frame_bgr, (rx, ry, rw, rh))
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
    text = pytesseract.image_to_string(thresh)

    cleaned = text.strip().replace('\n', ' ')
    match = re.search(r"Round\s*([0-9|Il]+).*?(Battle|Deploy)", cleaned, re.IGNORECASE)

    if match:
        raw_num = match.group(1)
        # Fix common OCR confusion: "|" → "1"
        round_num = int(raw_num.replace('|', '1'))
        phase = match.group(2).lower()
        result = (round_num, phase)
    else:
        result = (None, None)
    return result

def is_game_over(img: ImgLike) -> Tuple[bool, bool]:
    frame_bgr = _as_bgr(img)
    H, W = frame_bgr.shape[:2]
    geom = load_geometry(END_GEOM_PATH)
    rx, ry, rw, rh = get_abs_rect(geom, "play_again", W, H)

    roi = _crop(frame_bgr, (rx, ry, rw, rh))
    hsv  = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (0, 0, 230), (180, 40, 255)) 
    res = 255 - mask

    template = cv2.imread("assets/templates/play_again.png", cv2.IMREAD_GRAYSCALE)
    score = float(cv2.matchTemplate(res, template, cv2.TM_CCOEFF_NORMED)[0][0])
    if score > 0.6:
        return True, True
    
    rx, ry, rw, rh = get_abs_rect(geom, "ok", W, H)

    roi = _crop(frame_bgr, (rx, ry, rw, rh))
    hsv  = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (0, 0, 230), (180, 40, 255)) 
    res = 255 - mask

    template = cv2.imread("assets/templates/ok.png", cv2.IMREAD_GRAYSCALE)
    score = float(cv2.matchTemplate(res, template, cv2.TM_CCOEFF_NORMED)[0][0])
    return score > 0.6, False


def get_state(img: ImgLike) -> GameState:
    mana, mana_conf = read_mana(img)
    health = read_health(img)
    round, phase = read_round_phase(img)
    timer = read_timer_percent(img)
    return GameState(
        mana=mana,
        mana_conf=mana_conf,
        health=health,
        timer=timer,
        round=round,
        phase=phase,
        cards=read_cards(img),
        game_over=is_game_over(img)
    )


def is_home_screen(img: ImgLike) -> bool:
    frame_bgr = _as_bgr(img)
    H, W = frame_bgr.shape[:2]
    geom = load_geometry(HOME_GEOM_PATH)
    rx, ry, rw, rh = get_abs_rect(geom, "battle_button", W, H)

    roi = _crop(frame_bgr, (rx, ry, rw, rh))
    hsv  = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (0, 0, 230), (180, 40, 255)) 
    res = 255 - mask
    
    template = cv2.imread("assets/templates/battle.png", cv2.IMREAD_GRAYSCALE)
    score = float(cv2.matchTemplate(res, template, cv2.TM_CCOEFF_NORMED)[0][0])
    return score > 0.6
   
def get_placement(img: ImgLike) -> int:
    frame_bgr = _as_bgr(img)
    H, W = frame_bgr.shape[:2]
    geom = load_geometry(END_GEOM_PATH)

    for i, placement in enumerate(geom["placement"]):
        check = placement["player_check"]
        x, y, w, h = check["x"], check["y"], check["w"], check["h"]
        rx, ry, rw, rh = _to_abs_rect((x, y, w, h), W, H)
        roi = _crop(frame_bgr, (rx, ry, rw, rh))

        # Convert to HSV and check for yellow hue range
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, (20, 100, 150), (40, 255, 255)) 
        frac = float(np.count_nonzero(mask)) / float(mask.size)

        if frac > 0.1:  
            return i + 1

    return 0 

def get_roboflow_prediction(img: ImgLike) -> Tuple[Optional[str], Optional[int]]:
    load_dotenv()
    ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_API_KEY")
    if not ROBOFLOW_API_KEY:
        print("Warning: ROBOFLOW_API_KEY not found in .env file. Skipping initial unit detection.")
        return None, None
    frame_bgr = _as_bgr(img)

    # Load card data to determine melee/ranged
    with open("cards.json", "r") as f:
        card_data = json.load(f)

    try:
        CLIENT = InferenceHTTPClient(
            api_url="https://serverless.roboflow.com",
            api_key=ROBOFLOW_API_KEY
        )
        result = CLIENT.infer(frame_bgr, model_id="mtcd-board-v3-snw8t/2")
        print("Results:", result)
    except Exception as e:
        print(f"Error during Roboflow inference: {e}")
        return None, None

    preds = result.get("predictions") or []
    if not preds:
        print("No Roboflow predictions found.")
        return None, None

    p = preds[0]
    label = CLASS_TO_LABEL.get(p.get("class", ""), p.get("class", ""))
    if label not in card_data:
        print(f"Warning: {label} not found in cards.json")
        return None, None

    # Determine unit type (melee/ranged)
    unit_type = card_data[label]["type"]

    FRONT_ROW_MIDDLE_TILE = 2  
    BACK_ROW_MIDDLE_TILE = 17   

    if unit_type == "melee":
        tile_index = FRONT_ROW_MIDDLE_TILE
        print(f"Detected melee unit {label}, assigning to front-row middle tile {tile_index}.")
    else:
        tile_index = BACK_ROW_MIDDLE_TILE
        print(f"Detected ranged unit {label}, assigning to back-row middle tile {tile_index}.")

    return label, tile_index



# CLI (quick manual test)
if __name__ == "__main__":
    import argparse, sys

    ap = argparse.ArgumentParser(description="Read mana and hand costs via template matching using geometry.json")
    ap.add_argument("image", type=str, help="Path to screenshot image")
    args = ap.parse_args()

    start = time.perf_counter()   # start high-res timer
    state = get_state(args.image)
    is_home = is_home_screen(args.image)
    # label, tile_index = get_roboflow_prediction(args.image)
    
    print(f"Mana: {state.mana} ({state.mana_conf:.2f}% confi)")
    print(f"Health: {state.health}")
    print(f"Timer: {getattr(state, 'timer', None)}%")
    print(f"Round: {state.round}")
    print(f"Phase: {state.phase}")
    print(f"Game over: {state.game_over[0]} Play Again: {state.game_over[1]}")

    for c in state.cards:
        if c.label:
            print(f"{c.label:15s} conf={c.conf:.2f} idx={c.idx} cost={c.cost} type={c.type} "
                f"traits=({c.trait1}, {c.trait2}) upg={c.upgradable}")
            
    # print(f"Label: {label}, Tile Index: {tile_index}")
    print(f"Is Home Screen: {is_home}")
    print(f"\n--- Runtime: {time.perf_counter() - start:.3f} seconds ---")

   



 