from pathlib import Path
import cv2

DIGIT_DIR = Path("../assets/templates")

for d in range(10):
    path = DIGIT_DIR / f"{d}.png"
 
    # Load image
    img = cv2.imread(f"../assets/digits_raw/{d}.png")

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
    cv2.imwrite(path, res)
    print(f"Saved: {path}  (shape={res.shape})")

print(f"\n Templates saved to {DIGIT_DIR.resolve()}")