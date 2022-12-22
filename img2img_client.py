import base64
import time
from pathlib import Path
import requests
from PIL import Image
from io import BytesIO

ENDPOINT = "http://localhost:50121/predict" # local
ENDPOINT = "https://iqjik-01gmwy9q97ajqnvn96n230c9w5.litng-ai-03.litng.ai/predict" # cloud

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
        "image": load_img("./assets/sketch-mountains-input.jpg"),
    })
    json = response.json()
    print(i, time.time() - start)
    if "image" in json:
        img = json["image"]
        img = base64.b64decode(img.encode("utf-8"))
        Path(f"response_{i}.png").write_bytes(img)
    else:
        raise Exception(json)