# !pip install 'git+https://github.com/Lightning-AI/stablediffusion.git@lit'
# !pip install 'git+https://github.com/Lightning-AI/DiffusionWithAutoscaler.git'
# !pip install 'git+https://github.com/Lightning-AI/LAI-API-Access-UI-Component.git'
# !curl https://raw.githubusercontent.com/Lightning-AI/stablediffusion/lit/configs/stable-diffusion/v1-inference.yaml -o v1-inference.yaml
import time

import lightning as L
import base64, io, ldm, torch
from diffusion_with_autoscaler import AutoScaler, BatchTextImage, BatchImage, Text, Image, TextImage


class Img2ImgDiffusionServer(L.app.components.PythonServer):
    def __init__(self, *args, **kwargs):
        super().__init__(
            input_type=BatchTextImage,
            output_type=BatchImage,
            *args,
            **kwargs,
        )

    def setup(self):
        # cmd = "curl -C - https://pl-public-data.s3.amazonaws.com/dream_stable_diffusion/v1-5-pruned-emaonly.ckpt -o v1-5-pruned-emaonly.ckpt"
        # os.system(cmd)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model = ldm.lightning.LightningStableImg2ImgDiffusion(
            config_path="v1-inference.yaml",
            checkpoint_path="../../stable-diffusion/v1-5-pruned-emaonly.ckpt",
            device=device,
        )

    def predict(self, requests):
        start = time.time()
        batch_size = len(requests.inputs)
        texts = [request.text for request in requests.inputs]
        images = [request.image for request in requests.inputs]
        images = self._model.predict_step((texts, images), batch_idx=0)
        results = []
        for image in images:
            buffer = io.BytesIO()
            image.save(buffer, format="PNG")
            image_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
            results.append(image_str)
        print(f"finish predicting with batch size {batch_size} in {time.time()- start} seconds")
        return BatchImage(outputs=[{"image": image_str} for image_str in results])

    def on_exit(self):
        del self._model
        torch.cuda.empty_cache()


component = AutoScaler(
    Img2ImgDiffusionServer,  # The component to scale
    cloud_compute=L.CloudCompute("gpu-rtx", disk_size=80),

    # autoscaler args
    min_replicas=1,
    max_replicas=3,
    endpoint="/predict",
    scale_out_interval=0,
    scale_in_interval=300,  # 30 minutes
    max_batch_size=6,
    timeout_batching=2,
    input_type=TextImage,
    output_type=Image,
)

app = L.LightningApp(component)
