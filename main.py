import cv2
import numpy as np
import json

# Load your geometry definitions
with open("geometry.json") as f:
    GEOM = json.load(f)

print("Board has", GEOM["board"]["rows"], "rows and", GEOM["board"]["cols"], "columns.")
