import os, subprocess
from pathlib import Path

ADB_DEVICE = os.environ.get("ADB_DEVICE", "127.0.0.1:5555")  # set your actual serial

def find_adb():
    """Find adb.exe via Windows 'where' and return as a WSL Path."""
    try:
        # Ask Windows for adb.exe path
        win_path = subprocess.check_output(
            ["cmd.exe", "/C", "where adb"], text=True, cwd="/mnt/c", stderr=subprocess.DEVNULL  
        ).splitlines()[0].strip()

        # Convert C:\path\to\adb.exe → /mnt/c/path/to/adb.exe
        wsl_path = subprocess.check_output(
            ["wslpath", "-a", win_path], text=True
        ).strip()

        return Path(wsl_path)
    except Exception as e:
        raise FileNotFoundError(f"Unable to locate adb via Windows: {e}")

def adb_cmd(*args):
    base = [str(find_adb())]
    if ADB_DEVICE:
        base += ["-s", ADB_DEVICE]
    return base + [str(a) for a in args]

def adb_screenshot(out_path):
    with open(out_path, "wb") as f:
        subprocess.run(adb_cmd("exec-out", "screencap", "-p"), stdout=f, check=True)

def adb_tap(x, y):
    subprocess.check_call(adb_cmd("shell", "input", "tap", x, y))

def adb_swipe(x1, y1, x2, y2, duration_ms=250):
    subprocess.check_call(adb_cmd("shell", "input", "swipe", x1, y1, x2, y2, duration_ms))

if __name__ == "__main__":
    print("[TEST] Tapping center…")
    adb_tap(360, 640)
    print("[OK] Tap sent")

    print("[TEST] Swiping up…")
    adb_swipe(360, 400, 360, 200, 500)
    print("[OK] Swipe sent")
