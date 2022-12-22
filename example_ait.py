import torch
import os, base64, io, ldm, time
from diffusers import EulerDiscreteScheduler
from diffusion_with_autoscaler import (
    CustomColdStartProxy,
    AutoScaler,
    BatchText,
    BatchImage,
    Text,
    Image,
    StableDiffusionAITPipeline,
)

# cmd = f"curl -C - https://lightning-example-public.s3.amazonaws.com/stable-diffusion-2-bs1-ait.tar.gz -o stable-diffusion-2-bs1-ait.tar.gz"
# os.system(cmd)
# cmd = f"tar xzvf stable-diffusion-2-bs1-ait.tar.gz"
# os.system(cmd)

# TODO: Use local file completely
scheduler = EulerDiscreteScheduler.from_pretrained("stabilityai/stable-diffusion-2", subfolder="scheduler")
pipe = StableDiffusionAITPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2",
    scheduler=scheduler,
    revision="fp16",
    torch_dtype=torch.float16,
).to("cuda")

prompt = ["a man holding a pot", "something is great"]
height = 512
width = 512

with torch.autocast("cuda"):
    image = pipe(prompt, height, width).images
    # if benchmark:
    #     t = benchmark_torch_function(10, pipe, prompt, height=height, width=width)
    #     print(f"sd e2e: {t} ms")

print(image)
import time
filename = f"example_ait_{time.time()}.png"
image.save(filename)
print(f"saved to {filename}")
