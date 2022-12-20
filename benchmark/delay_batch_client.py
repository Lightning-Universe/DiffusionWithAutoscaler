import time
import base64
from pathlib import Path
import aiohttp
import asyncio

ENDPOINT = "https://epuji-01gmqhyzq3ve495v1rnz6s505c.litng-ai-03.litng.ai/predict"
TEXT = "A portrait of a person looking away from the camera"

async def async_request(idx):
    print(f"Starting {idx} ..")
    start = time.time()
    async with aiohttp.ClientSession() as session:
        async with session.post(ENDPOINT, json={"text": TEXT}) as result:
            response = await result.json()
            img = response["image"]
            img = base64.b64decode(img.encode("utf-8"))
            Path(f"response_{idx}.png").write_bytes(img)
    return time.time() - start

def simple_ramp_function(num_requests: int = 2, concurrent_requests: list[int] = [1, 2, 3, 4, 5, 6, 7, 8]):
    requests = []
    for step in concurrent_requests:
        requests += [step] * num_requests
    return requests

def run_benchmark(num_requests):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    counter = 0
    for num_request in num_requests:
        coros = []
        for _ in range(num_request):
            coros.append(async_request(counter))
            counter += 1

        results = loop.run_until_complete(asyncio.gather(*coros))
        print(results)


if __name__ == "__main__":
    run_benchmark(simple_ramp_function())