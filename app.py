# !pip install lightning_api_access
# !pip install 'git+https://github.com/Lightning-AI/stablediffusion.git@lit'
# !curl https://raw.githubusercontent.com/Lightning-AI/stablediffusion/main/configs/stable-diffusion/v2-inference-v.yaml -o v2-inference-v.yaml
import time

import lightning as L
import torch
import os, base64, io, ldm

from autoscaler import AutoScaler
from cold_start_proxy import CustomColdStartProxy
from datatypes import BatchText, BatchResponse, Text, Image

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"


class DiffusionServer(L.app.components.PythonServer):
    def __init__(self, *args, **kwargs):
        super().__init__(
            input_type=BatchText,
            output_type=BatchResponse,
            *args,
            **kwargs,
        )

    def setup(self):
        cmd = "curl -C - https://pl-public-data.s3.amazonaws.com/dream_stable_diffusion/768-v-ema.ckpt -o 768-v-ema.ckpt"
        os.system(cmd)

        device = "cuda" if torch.cuda.is_available() else "cpu"

        self._model = ldm.lightning.LightningStableDiffusion(
            config_path="v2-inference-v.yaml",
            checkpoint_path="768-v-ema.ckpt",
            device=device,
        ).to(device)

        # TODO - float16 and no grad

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def predict(self, requests):
        start = time.time()
        batch_size = len(requests.inputs)
        texts = [request.text for request in requests.inputs]

        images = self._model.predict_step(
            prompts=texts,
            batch_idx=0,  # or whatever
        )

        results = []
        for image in images:
            buffer = io.BytesIO()
            image.save(buffer, format="PNG")
            image_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
            results.append(image_str)

        print(f"finish predicting with batch size {batch_size} in {time.time() - start} seconds")
        return BatchResponse(outputs=[{"image": image_str} for image_str in results])


component = AutoScaler(
    DiffusionServer,  # The component to scale
    cloud_compute=L.CloudCompute("gpu-rtx", disk_size=80),

    # autoscaler args
    min_replicas=1,
    max_replicas=3,
    endpoint="/predict",
    autoscale_up_interval=0,
    autoscale_down_interval=1800,  # 30 minutes
    max_batch_size=8,
    timeout_batching=2,
    input_type=Text,
    output_type=Image,
    cold_start_proxy=CustomColdStartProxy(),
)

app = L.LightningApp(component)
