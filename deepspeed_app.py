# !pip install deepspeed==0.7.5 deepspeed-mii==0.0.3 diffusers==0.7.1 transformers==4.24.0 triton==2.0.0.dev20221005
# !pip install 'git+https://github.com/Lightning-AI/LAI-API-Access-UI-Component.git'
# !pip install 'git+https://github.com/Lightning-AI/DiffusionWithAutoscaler.git@debugging-deepspeed'

import lightning as L
import os, base64, io, torch, diffusers, deepspeed
from diffusion_with_autoscaler import AutoScaler, BatchText, BatchImage, Text, Image, PreemptibleRollout


class DiffusionServer(L.app.components.PythonServer):
    def __init__(self, *args, **kwargs):
        super().__init__(
            input_type=BatchText,
            output_type=BatchImage,
            *args,
            **kwargs,
        )

    def setup(self):
        hf_auth_key = os.getenv("HF_AUTH_KEY")
        if not hf_auth_key:
            raise ValueError("HF_AUTH_KEY is not set")
        pipe = diffusers.StableDiffusionPipeline.from_pretrained(
            "CompVis/stable-diffusion-v1-4",
            use_auth_token=hf_auth_key,
            torch_dtype=torch.float16,
            revision="fp16")
        self._model = deepspeed.init_inference(pipe.to("cuda"), dtype=torch.float16)

    def predict(self, requests):
        texts = [request.text for request in requests.inputs]
        resp = self._model(texts, num_inference_steps=30)
        results = []
        for image in resp[0]:
            buffer = io.BytesIO()
            image.save(buffer, format="PNG")
            image_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
            results.append(image_str)
        return BatchImage(outputs=[{"image": image_str} for image_str in results])


component = AutoScaler(
    DiffusionServer,  # The component to scale
    cloud_compute=L.CloudCompute("gpu-rtx", disk_size=80, preemptible=True),

    strategy=PreemptibleRollout(interval=60),
    # autoscaler args
    min_replicas=1,
    max_replicas=1,
    endpoint="/predict",
    scale_out_interval=0,
    scale_in_interval=600,
    max_batch_size=6,
    timeout_batching=0.3,
    input_type=Text,
    output_type=Image
)

app = L.LightningApp(component)