#!/usr/bin/env python3
"""
controller.py — input/control API for Merge Tactics

Primary API:
    - drag(src: Slot, dst: Slot, duration_ms: int = 120) -> None

Where a Slot can be:
    - Hand(idx: int)
    - Bench(idx: int)
    - Board(row: int, col: int)
    - Pixel(x: int, y: int)    # absolute screen pixels


    python controller.py --from hand:0 --to board:2,3
    python controller.py --from board:1,1 --to hand:0          # example "sell"
    python controller.py --from bench:4 --to board:0,2
    python controller.py --from px:200,1200 --to px:540,400
"""

from __future__ import annotations
import json
import re
import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple, Union

from adb_wrap import adb_swipe, adb_tap

# Geometry / config paths
GEOM_PATH = Path("geometry.json")
END_GEOM_PATH = Path("end-screen-geometry.json")
HOME_GEOM_PATH = Path("home-screen-geometry.json")

# Slot types
@dataclass(frozen=True)
class Hand:
    idx: int

@dataclass(frozen=True)
class Bench:
    idx: int

@dataclass(frozen=True)
class Board:
    row: int
    col: int

@dataclass(frozen=True)
class Pixel:
    x: int
    y: int

@dataclass(frozen=True)
class Button:
    type: str

Slot = Union[Hand, Bench, Board, Pixel, Button]

# Geometry helpers
def _norm_to_px(val: float, total: int) -> int:
    return int(round(val * total))

def _center_wh_to_rect_px(cx: float, cy: float, w: float, h: float, W: int, H: int) -> Tuple[int,int,int,int]:
    px_w = _norm_to_px(w, W)
    px_h = _norm_to_px(h, H)
    px_cx = _norm_to_px(cx, W)
    px_cy = _norm_to_px(cy, H)
    return (px_cx - px_w // 2, px_cy - px_h // 2, px_w, px_h)

def _rect_center(r: Tuple[int,int,int,int]) -> Tuple[int,int]:
    x,y,w,h = r
    return (x + w // 2, y + h // 2)

def _load_geom(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing geometry file: {path}")
    return json.loads(path.read_text())

# Slot → pixel center
def _slot_center(slot: Slot, geom: Dict[str, Any]) -> Tuple[int, int]:
    W, H = geom["resolution_px"]

    if isinstance(slot, Pixel):
        # Already absolute pixels
        return (int(slot.x), int(slot.y))
    
    if isinstance(slot, Button):
        if slot.type == "play_again":
            button = geom.get("play_again_center") 
            if not isinstance(button, dict):
                raise ValueError("geometry.json missing 'Play Again' object")
            center = _center_wh_to_rect_px(button["cx"], button["cy"], button["w"], button["h"], W, H)
            return _rect_center(center)
        elif slot.type == "return_home":
            button = geom.get("ok_center") 
            if not isinstance(button, dict):
                raise ValueError("geometry.json missing 'OK' object")
            center = _center_wh_to_rect_px(button["cx"], button["cy"], button["w"], button["h"], W, H)
            return _rect_center(center)
        elif slot.type == "start_battle":
            button = geom.get("battle_button_center")
            if not isinstance(button, dict):
                raise ValueError("geometry.json missing 'battle button' object")
            center = _center_wh_to_rect_px(button["cx"], button["cy"], button["w"], button["h"], W, H)
            return _rect_center(center)
        
        raise ValueError(f"Unsupported button type: {slot.type}")


    if isinstance(slot, Hand):
        hand = geom.get("hand")
        if not isinstance(hand, list):
            raise ValueError("geometry.json missing 'hand' list.")
        if not (0 <= slot.idx < len(hand)):
            raise IndexError(f"hand index {slot.idx} out of range (have {len(hand)}).")
        h = hand[slot.idx]
        r = _center_wh_to_rect_px(h["cx"], h["cy"], h["w"], h["h"], W, H)
        return _rect_center(r)

    if isinstance(slot, Bench):
        bench = geom.get("bench")
        if not isinstance(bench, list):
            raise ValueError("geometry.json missing 'bench' list.")
        if not (0 <= slot.idx < len(bench)):
            raise IndexError(f"bench index {slot.idx} out of range (have {len(bench)}).")
        b = bench[slot.idx]
        cx = b["x"] + b["w"] / 2.0
        cy = b["y"] + b["h"] / 2.0
        r = _center_wh_to_rect_px(cx, cy, b["w"], b["h"], W, H)
        return _rect_center(r)

    if isinstance(slot, Board):
        board = geom.get("board")
        if not isinstance(board, dict):
            raise ValueError("geometry.json missing 'board' object.")
        rows = board["rows"]; cols = board["cols"]
        tiles = board["tiles"]
        if not (0 <= slot.row < rows and 0 <= slot.col < cols):
            raise IndexError(f"board ({slot.row},{slot.col}) out of range; rows={rows}, cols={cols}.")
        t = tiles[slot.row * cols + slot.col]
        r = _center_wh_to_rect_px(t["cx"], t["cy"], t["w"], t["h"], W, H)
        return _rect_center(r)

    raise TypeError(f"Unsupported slot type: {slot}")

# Public API
def drag(src: Slot, dst: Slot, duration_ms: int = 280) -> None:
    """
    Drag from the center of `src` to the center of `dst` using adb_swipe.
    Dragging TO a Hand slot is interpreted by the game as 'sell'.
    """
    geom = _load_geom(GEOM_PATH)
    x1, y1 = _slot_center(src, geom)
    x2, y2 = _slot_center(dst, geom)
    adb_swipe(x1, y1, x2, y2, duration_ms)

def quick_buy(hand: Hand) -> None:
    geom = _load_geom(GEOM_PATH)
    x1, y1 = _slot_center(hand, geom)
    adb_tap(x1, y1)

def play_again() -> None:
    geom = _load_geom(END_GEOM_PATH)
    x1, y1 = _slot_center(Button("play_again"), geom)
    adb_tap(x1, y1)

def return_home() -> None:
    geom = _load_geom(END_GEOM_PATH)
    x1, y1 = _slot_center(Button("return_home"), geom)
    adb_tap(x1, y1)

def start_battle():
    geom = _load_geom(HOME_GEOM_PATH)
    x1, y1 = _slot_center(Button("start_battle"), geom)
    adb_tap(x1, y1)


# Minimal CLI for quick manual tests
_KIND_PATTERNS = {
    "hand":  re.compile(r"^hand:(\d+)$", re.I),
    "bench": re.compile(r"^bench:(\d+)$", re.I),
    "board": re.compile(r"^board:(\d+),(\d+)$", re.I),
    "px":    re.compile(r"^px:(\d+),(\d+)$", re.I),
}

def _parse_slot(text: str) -> Slot:
    text = text.strip()
    m = _KIND_PATTERNS["hand"].match(text)
    if m: return Hand(int(m.group(1)))
    m = _KIND_PATTERNS["bench"].match(text)
    if m: return Bench(int(m.group(1)))
    m = _KIND_PATTERNS["board"].match(text)
    if m: return Board(int(m.group(1)), int(m.group(2)))
    m = _KIND_PATTERNS["px"].match(text)
    if m: return Pixel(int(m.group(1)), int(m.group(2)))
    raise ValueError(f"Could not parse slot spec: '{text}'. "
                     "Examples: hand:0  bench:3  board:2,4  px:540,960")

def _main():
    ap = argparse.ArgumentParser(description="General drag API CLI")
    ap.add_argument("--from", dest="src", required=True, help="hand:i | bench:i | board:r,c | px:x,y")
    ap.add_argument("--to",   dest="dst", required=True, help="hand:i | bench:i | board:r,c | px:x,y")
    ap.add_argument("--duration", type=int, default=120, help="swipe duration ms")
    ap.add_argument("--dry", action="store_true", help="print only; do not send swipe")
    args = ap.parse_args()

    src = _parse_slot(args.src)
    dst = _parse_slot(args.dst)

    geom = _load_geom(GEOM_PATH)
    x1, y1 = _slot_center(src, geom)
    x2, y2 = _slot_center(dst, geom)

    print(f"[info] drag {args.src} ({x1},{y1}) -> {args.dst} ({x2},{y2})  duration={args.duration}ms")

    if args.dry:
        print("[dry-run] not sending swipe")
        return

    adb_swipe(x1, y1, x2, y2, args.duration)
    print("[ok] swipe sent")

if __name__ == "__main__":
    _main()
