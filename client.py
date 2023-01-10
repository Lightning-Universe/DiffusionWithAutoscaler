import base64
import time
from pathlib import Path
import requests

for i in range(100):
    start = time.time()
    response = requests.post('https://buaac-01gpc0bjpwqxkk143xmr73dcs2.litng-ai-03.litng.ai/predict', json={
        "text": "astronaut riding a horse, digital art, epic lighting, highly-detailed masterpiece trending HQ"
    })

    img = response.json()["image"]
    img = base64.b64decode(img.encode("utf-8"))
    Path(f"response_{i}.png").write_bytes(img)
    print(i, time.time() - start)
