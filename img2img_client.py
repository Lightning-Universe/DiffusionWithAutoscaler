import base64
import time
from pathlib import Path
import requests

ENDPOINT = "http://localhost:50121/predict"  # local
ENDPOINT = "https://iqjik-01gmwy9q97ajqnvn96n230c9w5.litng-ai-03.litng.ai/predict"  # cloud

url = "https://raw.githubusercontent.com/CompVis/stable-diffusion/main/assets/stable-samples/img2img/sketch-mountains-input.jpg"
image = requests.get(url).content
image = base64.b64encode(image).decode("ascii")

for i in range(100):
    start = time.time()
    response = requests.post(
        ENDPOINT,
        json={
            "text": "A fantasy landscape, trending on artstation",
            "image": image,
        },
    )
    json = response.json()
    print(i, time.time() - start)
    if "image" in json:
        img = json["image"]
        img = base64.b64decode(img.encode("utf-8"))
        Path(f"response_{i}.png").write_bytes(img)
    else:
        raise Exception(json)
