import os, subprocess
from pathlib import Path


ADB_BIN = Path("/mnt/c/Users/Artemis Tran/Downloads/platform-tools-latest-windows/platform-tools/adb.exe")
ADB_DEVICE = os.environ.get("ADB_DEVICE", "127.0.0.1:5555")  # set your actual serial

def adb_cmd(*args):
    base = [str(ADB_BIN)]
    if ADB_DEVICE:
        base += ["-s", ADB_DEVICE]
    return base + [str(a) for a in args]

def adb_screenshot(out_path):
    # allow str path or Path
    with open(out_path, "wb") as f:
        subprocess.run(adb_cmd("exec-out", "screencap", "-p"), stdout=f, check=True)

def adb_tap(x, y):
    subprocess.check_call(adb_cmd("shell", "input", "tap", x, y))

def adb_swipe(x1, y1, x2, y2, duration_ms=250):
    subprocess.check_call(adb_cmd("shell", "input", "swipe", x1, y1, x2, y2, duration_ms))

if __name__ == "__main__":
    # Sanity: confirm we can invoke adb
    print("[TEST] ADB BIN:", ADB_BIN)
    subprocess.check_call([str(ADB_BIN), "version"])

    print("[TEST] Tapping center…")
    adb_tap(360, 640)
    print("[OK] Tap sent")

    print("[TEST] Swiping up…")
    adb_swipe(360, 400, 360, 200, 500)
    print("[OK] Swipe sent")
