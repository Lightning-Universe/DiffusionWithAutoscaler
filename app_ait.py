# !pip install diffusers transformers
# !pip install 'git+https://github.com/Lightning-AI/stablediffusion.git@lit'
# !pip install 'git+https://github.com/Lightning-AI/DiffusionWithAutoscaler.git'
# !pip install 'git+https://github.com/Lightning-AI/LAI-API-Access-UI-Component.git'
# !curl https://raw.githubusercontent.com/Lightning-AI/stablediffusion/main/configs/stable-diffusion/v2-inference-v.yaml -o v2-inference-v.yaml
import lightning as L

import torch
import os, base64, io, ldm, time

from diffusion_with_autoscaler import (
    CustomColdStartProxy,
    AutoScaler,
    BatchText,
    BatchImage,
    Text,
    Image,
    StableDiffusionAITPipeline,
)

PROXY_URL = "https://ulhcn-01gd3c9epmk5xj2y9a9jrrvgt8.litng-ai-03.litng.ai/api/predict"


class DiffusionServer(L.app.components.PythonServer):
    def __init__(self, *args, **kwargs):
        super().__init__(
            input_type=BatchText,
            output_type=BatchImage,
            *args,
            **kwargs,
        )

    def setup(self):
        cmd = "curl -C - https://lightning-example-public.s3.amazonaws.com/stable-diffusion-2-bs1-ait.tar.gz -o stable-diffusion-2-bs1-ait.tar.gz"
        os.system(cmd)
        cmd = "tar xzvf stable-diffusion-2-bs1-ait.tar.gz"
        os.system(cmd)
        cmd = "ln -s stable-diffusion-2-bs1-ait tmp"
        os.system(cmd)

        # TODO: Switch to non-AIT model when not on cuda device
        # TODO: Use local file completely instead of partially relying on "stabilityai/stable-diffusion-2"
        scheduler = EulerDiscreteScheduler.from_pretrained("stabilityai/stable-diffusion-2", subfolder="scheduler")
        self._model = StableDiffusionAITPipeline.from_pretrained(
            "stabilityai/stable-diffusion-2",
            scheduler=scheduler,
            revision="fp16",
            torch_dtype=torch.float16,
        ).to("cuda")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def predict(self, requests):
        start = time.time()
        batch_size = len(requests.inputs)
        texts = [request.text for request in requests.inputs]
        # images = self._model(prompt=texts, 512, 512).images
        images = self._model(prompt=texts[0], 512, 512).images
        results = []
        for image in images:
            buffer = io.BytesIO()
            image.save(buffer, format="PNG")
            image_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
            results.append(image_str)
        print(
            f"finish predicting with batch size {batch_size} in {time.time() - start} seconds"
        )
        return BatchImage(outputs=[{"image": image_str} for image_str in results])


component = AutoScaler(
    DiffusionServer,  # The component to scale
    cloud_compute=L.CloudCompute("gpu-rtx", disk_size=80),
    # autoscaler args
    min_replicas=1,
    max_replicas=3,
    endpoint="/predict",
    scale_out_interval=0,
    scale_in_interval=300,  # 30 minutes
    max_batch_size=8,
    timeout_batching=2,
    input_type=Text,
    output_type=Image,
    cold_start_proxy=CustomColdStartProxy(proxy_url=PROXY_URL),
)

app = L.LightningApp(component)
