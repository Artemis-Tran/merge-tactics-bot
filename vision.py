#!/usr/bin/env python3
"""
vision.py — screen capture module for Merge Tactics

Provides:
  - Vision.start_stream(fps=5)
  - Vision.stop_stream()
  - Vision.get_frame() -> np.ndarray | None
  - Vision.capture_frame() -> np.ndarray
  - Vision.capture_after_action(action_fn, *, settle_ms=180) -> np.ndarray

All functions return frames as BGR np.ndarrays.
"""

from __future__ import annotations
import subprocess, threading, time
from typing import Optional
import numpy as np
import cv2

from adb_wrap import adb_screenshot
 
def _decode_png_to_bgr(png_bytes: bytes) -> np.ndarray:
    """Decode PNG bytes into an OpenCV BGR frame."""
    arr = np.frombuffer(png_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError("Failed to decode screencap from ADB.")
    return img

class _StreamBackend:
    """
    Simple threaded stream that polls `adb exec-out screencap -p` at low FPS.
    """

    def __init__(self, fps: int = 5):
        self.target_fps = fps
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._frame: Optional[np.ndarray] = None

    def start(self):
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=1.0)

    def set_fps(self, fps: int):
        self.target_fps = max(1, int(fps))

    def latest_frame(self) -> Optional[np.ndarray]:
        with self._lock:
            if self._frame is None:
                return None
            return self._frame.copy()

    def _loop(self):
        delay = 1.0 / self.target_fps
        while not self._stop.is_set():
            t0 = time.time()
            try:
                png = adb_screenshot()
                img = _decode_png_to_bgr(png)
                with self._lock:
                    self._frame = img
            except Exception:
                pass
            # adaptive sleep to hold target FPS
            elapsed = time.time() - t0
            time.sleep(max(0.0, delay - elapsed))

class Vision:
    def __init__(self, fps: int = 5):
        self.stream = _StreamBackend(fps=fps)

    #  Stream control 
    def start_stream(self, fps: int = 5):
        self.stream.set_fps(fps)
        self.stream.start()

    def stop_stream(self):
        self.stream.stop()

    def get_frame(self) -> Optional[np.ndarray]:
        """Return latest streamed frame as np.ndarray (BGR), or None if not ready."""
        return self.stream.latest_frame()

    # Single deterministic captures 
    def capture_frame(self) -> np.ndarray:
        """Take one fresh screencap and return it as np.ndarray (BGR)."""
        png = adb_screenshot()
        return _decode_png_to_bgr(png)

    def capture_after_action(self, action_fn, *, settle_ms: int = 180) -> np.ndarray:
        """
        Run an action (e.g., drag), wait settle_ms for UI to update, then capture frame.
        """
        action_fn()
        time.sleep(settle_ms / 1000.0)
        return self.capture_frame()
