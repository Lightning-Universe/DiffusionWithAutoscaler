import base64
from pathlib import Path
import requests
import threading


def req_and_write():
    response = requests.post('https://mcegy-01gmwahccg6t8e9ew84095xtq4.litng-ai-03.litng.ai/predict', json={
        "text": "A portrait of a person looking away from the camera"
    })
    img = response.json()["image"]
    img = base64.b64decode(img.encode("utf-8"))
    Path("response.png").write_bytes(img)


threads = []
for i in range(6):
    t = threading.Thread(target=req_and_write)
    threads.append(t)
    t.start()
print("started all threads")
for t in threads:
    t.join()
