import base64
import time
from pathlib import Path
import requests


def req_and_write(index):
    start = time.time()
    response = requests.post("https://vkrdr-01gp18trs3dcr2ts95h2b3w6vv.litng-ai-03.litng.ai/predict", json={
        "text": "astronaut riding a horse, digital art, epic lighting, highly-detailed masterpiece trending HQ"
    })
    end = time.time()
    try:
        img = response.json()["image"]
    except Exception:
        print("index {} failed".format(index))
        print(response.text)
        print("=======================================\n\n")
    else:
        img = base64.b64decode(img.encode("utf-8"))
        Path(f"response{index}.png").write_bytes(img)
        print("index {} success in {}".format(index, end - start))


req_and_write(0)
