import base64
import time
from pathlib import Path
import requests

for i in range(100):
    start = time.time()
    response = requests.post('https://ulhcn-01gd3c9epmk5xj2y9a9jrrvgt8.litng-ai-03.litng.ai/api/predict', json={
        "prompt": "A portrait of a person looking away from the camera"
    })
    print(response.text)
    img = response.json()["image"]
    img = base64.b64decode(img.encode("utf-8"))
    Path(f"response_{i}.png").write_bytes(img)
    print(i, time.time() - start)
