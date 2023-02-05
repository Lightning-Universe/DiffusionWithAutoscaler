import base64
import time
from pathlib import Path
import requests
import threading


def req_and_write(index):
    start = time.time()
    response = requests.post(
        "https://zqihw-01gnyzn7w79g5an1074fdv9tnb.litng-ai-03.litng.ai/predict",
        json={"text": "astronaut riding a horse, digital art, epic lighting, highly-detailed masterpiece trending HQ"},
    )
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


i = 0
while True:
    print("")
    for _ in range(1):
        t = threading.Thread(target=req_and_write, args=(i,))
        t.start()
        i += 1
    time.sleep(10)
