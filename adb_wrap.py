import os, subprocess
from pathlib import Path

ADB_DEVICE = os.environ.get("ADB_DEVICE", "127.0.0.1:5555")  # set your actual serial
_adb_connected_once = False

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
    
def ensure_adb_connected():
    """
    For TCP/IP devices (e.g. 127.0.0.1:5555), automatically run
    `adb connect ADB_DEVICE` once per process.
    USB devices are ignored (no connect needed).
    """
    global _adb_connected_once

    if _adb_connected_once:
        return

    if not (ADB_DEVICE and ":" in ADB_DEVICE):
        _adb_connected_once = True
        return

    try:
        adb_path = str(find_adb())
        print(f"[adb] Auto-connecting to {ADB_DEVICE}...")
        # IMPORTANT: `adb connect` must NOT use `-s`
        subprocess.run([adb_path, "connect", ADB_DEVICE], check=False)
    except Exception as exc:
        print(f"[adb] Auto-connect failed: {exc}")
    finally:
        _adb_connected_once = True


def adb_cmd(*args):
    ensure_adb_connected()

    base = [str(find_adb())]
    if ADB_DEVICE:
        base += ["-s", ADB_DEVICE]
    return base + [str(a) for a in args]

def adb_screenshot():
    p = subprocess.run(adb_cmd("exec-out", "screencap", "-p"), stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    return p.stdout

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
