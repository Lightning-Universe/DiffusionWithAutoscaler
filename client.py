
import base64
import time
from pathlib import Path
import requests
import threading



def req_and_write(index):
    start = time.time()
    response = requests.post(
        'http://127.0.0.1:6001/predict', 
        json={
            "text": "astronaut riding a horse, digital art, epic lighting, highly-detailed masterpiece trending HQ"
        },
        # json={
        #     "inputs": [{
        #         "text": "astronaut riding a horse, digital art, epic lighting, highly-detailed masterpiece trending HQ"
        #     }]
        # },
        headers = {
            "accept": "application/json",
            "Content-Type": "application/json",
        }
    )
    end = time.time()
    print(end - start)
    try:
        img = response.json()["image"]
    except Exception:
        # print("index {} failed".format(index))
        #print(response.text)
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
