import base64
import time
from pathlib import Path
import requests
from PIL import Image
from io import BytesIO

ENDPOINT = "http://localhost:50121/predict"
# ENDPOINT = "https://bmdla-01gmr4je8s8zgnzpv8vyxpwn61.litng-ai-03.litng.ai/predict"

def load_img(path):
    image = Image.open(path).convert("RGB")
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)
    encoded = buffer.getvalue()
    return base64.b64encode(encoded).decode("ascii")

for i in range(100):
    start = time.time()
    response = requests.post(ENDPOINT, json={
        "text": "A fantasy landscape, trending on artstation",
        "image": load_img("../assets/sketch-mountains-input.jpg"),
    })
    json = response.json()
    if "image" in json:
        img = json["image"]
        img = base64.b64decode(img.encode("utf-8"))
        Path(f"response_{i}.png").write_bytes(img)
        print(i, time.time() - start)
    else:
        raise Exception(json)
