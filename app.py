# !pip install lightning_api_access
# !pip install 'git+https://github.com/Lightning-AI/stablediffusion.git@lit'
# !curl https://raw.githubusercontent.com/Lightning-AI/stablediffusion/main/configs/stable-diffusion/v2-inference-v.yaml -o v2-inference-v.yaml
import lightning as L
import torch, torch.utils.data as data
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

        precision = 16 if torch.cuda.is_available() else 32
        self._trainer = L.Trainer(
            accelerator="auto",
            devices=1,
            precision=precision,
            enable_progress_bar=False,
        )

        self._model = ldm.lightning.LightningStableDiffusion(
            config_path="v2-inference-v.yaml",
            checkpoint_path="768-v-ema.ckpt",
            device=self._trainer.strategy.root_device.type,
        )

        if torch.cuda.is_available():
            self._model = self._model.to(torch.float16)
            torch.cuda.empty_cache()

    def predict(self, requests):
        batch_size = len(requests.inputs)
        print(f"predicting with batch size {batch_size}")
        texts = []
        for request in requests.inputs:
            print("INFO: ", request.text)
            texts.append(request.text)

        images = self._trainer.predict(
            self._model,
            data.DataLoader(ldm.lightning.PromptDataset(texts), batch_size=batch_size),
        )[0]
        results = []
        for image in images:
            buffer = io.BytesIO()
            image.save(buffer, format="PNG")
            image_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
            results.append(image_str)

        return BatchResponse(outputs=[{"image": image_str} for image_str in results])


component = AutoScaler(
    # work cls and args
    DiffusionServer,
    cloud_compute=L.CloudCompute("gpu-rtx", disk_size=80),
    # autoscaler args
    min_replicas=0,
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
