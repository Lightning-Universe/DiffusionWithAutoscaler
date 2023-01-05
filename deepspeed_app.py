# !pip install deepspeed==0.7.5 deepspeed-mii==0.0.3 diffusers==0.7.1 transformers==4.24.0 triton==2.0.0.dev20221005
# !pip install 'git+https://github.com/Lightning-AI/DiffusionWithAutoscaler.git'

import lightning as L
import os, base64, io, torch, time, diffusers, deepspeed
from diffusion_with_autoscaler import AutoScaler, BatchText, BatchImage, Text, Image


class DiffusionServer(L.app.components.PythonServer):
    def __init__(self, *args, **kwargs):
        super().__init__(
            input_type=BatchText,
            output_type=BatchImage,
            *args,
            **kwargs,
        )

    def setup(self):
        hf_auth_key = "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"
        pipe = diffusers.StableDiffusionPipeline.from_pretrained(
            "CompVis/stable-diffusion-v1-4",
            use_auth_token=hf_auth_key,
            torch_dtype=torch.float16,
            revision="fp16")
        self._model = deepspeed.init_inference(pipe.to("cuda"), dtype=torch.float16)

    def predict(self, requests):
        t1 = time.time()
        text = requests.inputs[0].text
        t2 = time.time()
        image = self._model(text, num_inference_steps=30).images[0]
        t3 = time.time()
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        image_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
        t4 = time.time()
        print(f"t1: {t2-t1}, t2: {t3-t2}, t3: {t4-t3}")
        print(f"total: {t4-t1}")
        return BatchImage(outputs=[{"image": image_str}])


component = AutoScaler(
    DiffusionServer,  # The component to scale
    cloud_compute=L.CloudCompute("gpu-rtx", disk_size=80),

    # autoscaler args
    min_replicas=1,
    max_replicas=1,
    endpoint="/predict",
    scale_out_interval=0,
    scale_in_interval=600,
    max_batch_size=1,
    timeout_batching=0.1,
    input_type=Text,
    output_type=Image
)

app = L.LightningApp(component)




"""
# from deepspeed.env_report import cli_main
# cli_main()

import time
import torch
import diffusers
import deepspeed


hf_auth_key = "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"
pipe = diffusers.StableDiffusionPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4",
    use_auth_token=hf_auth_key,
    torch_dtype=torch.float16,
    revision="fp16").to("cuda")


s = time.time()
image = pipe("real life portrait of a person looking away from camera", num_inference_steps=30).images[0]
print(time.time() - s)
image.save("image.png")



# deep speed
inf_config = {"dtype": "fp16"}
engine = deepspeed.init_inference(pipe, config=inf_config)

s = time.time()
image = engine("real life portrait of a person looking away from camera", num_inference_steps=30).images[0]
print(time.time() - s)
image.save("image.png")

import mii

mii_config = {
    "dtype": "fp16",
    "hf_auth_token": "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"
}

mii.deploy(task='text-to-image',
           model="CompVis/stable-diffusion-v1-4",
           deployment_name="sd_deploy",
           mii_config=mii_config)

"""