import time
import base64
from pathlib import Path
import aiohttp
import numpy as np
import asyncio
from PIL import Image
from io import BytesIO

def load_img(path):
    image = Image.open(path).convert("RGB")
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)
    encoded = buffer.getvalue()
    return base64.b64encode(encoded).decode("ascii")

ENDPOINT = "http://localhost:50177/predict" # local
ENDPOINT = "https://iqjik-01gmwy9q97ajqnvn96n230c9w5.litng-ai-03.litng.ai/predict" # cloud
TEXT = "A fantasy landscape, trending on artstation"
IMAGE = load_img("./assets/sketch-mountains-input.jpg")


async def async_request(counter, sleep = 0):
    begin = time.time()
    await asyncio.sleep(sleep)
    print(f"Starting {counter} .. after {sleep}")
    start = time.time()
    async with aiohttp.ClientSession() as session:
        async with session.post(ENDPOINT, json={"text": TEXT, "image": IMAGE}) as result:
            response = await result.json()
            end = time.time()
            print(counter, end - start, end - begin)
            if "image"in response:
                img = response["image"]
                img = base64.b64decode(img.encode("utf-8"))
                Path(f"response_{counter}.png").write_bytes(img)
            else:
                raise Exception(response)
    return end - start


def compute_stats(inference_times):
    return {
        "avg": np.mean(inference_times),
        "median": np.median(inference_times),
        "p95": np.percentile(inference_times, 95),
    }


def run_benchmark(sleep_formats):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    counter = 0
    coros = []
    for sleep in sleep_formats:
        coros.append(async_request(counter=counter, sleep=sleep))
        counter += 1

    t0 = time.time()
    results = loop.run_until_complete(asyncio.gather(*coros))
    print(compute_stats(results))
    print("Overall Time", time.time() - t0)


def two_requests_every_N_seconds(seconds):
    requests = []
    for idx in range(0, 120, seconds):
        requests.append(idx)
        requests.append(idx)
    return requests


if __name__ == "__main__":
    run_benchmark(two_requests_every_N_seconds(9))