# !pip install 'git+https://github.com/Lightning-AI/LAI-API-Access-UI-Component.git'
# !curl https://raw.githubusercontent.com/Lightning-AI/stablediffusion/lit/configs/stable-diffusion/v1-inference.yaml -o v1-inference.yaml
import lightning as L
import os, base64, io
from diffusion_with_autoscaler import AutoScaler, BatchText, BatchImage, Text, Image

PROXY_URL = "https://ulhcn-01gd3c9epmk5xj2y9a9jrrvgt8.litng-ai-03.litng.ai/api/predict"


class StableDiffusionConfig(L.BuildConfig):

    def build_commands(self):
        return [
            "TORCH_CUDA_ARCH_LIST=7.5 pip install 'git+https://github.com/Lightning-AI/stablediffusion.git@add_support_for_flash_attention'"
        ]


class DiffusionServer(L.app.components.PythonServer):
    def __init__(self, *args, **kwargs):
        super().__init__(
            input_type=BatchText,
            output_type=BatchImage,
            *args,
            cloud_build_config=StableDiffusionConfig(),
            **kwargs,
        )
            
    def setup(self):
        cmd = "curl -C - https://pl-public-data.s3.amazonaws.com/dream_stable_diffusion/v1-5-pruned-emaonly.ckpt -o v1-5-pruned-emaonly.ckpt"
        os.system(cmd)

        import ldm, torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model = ldm.lightning.LightningStableDiffusion(
            config_path="v1-inference.yaml",
            checkpoint_path="v1-5-pruned-emaonly.ckpt",
            device=device,
            fp16=True, # Supported on GPU, skipped otherwise.
            use_deepspeed=True, # Supported on Ampere and RTX, skipped otherwise.
            use_triton_attention=False, # Whether to use OpenAI Triton kernels
            enable_cuda_graph=True,
            steps=30,         
        )

    def predict(self, requests):
        texts = [request.text for request in requests.inputs]
        images = self._model.predict_step(prompts=texts, batch_idx=0)
        results = []
        for image in images:
            buffer = io.BytesIO()
            image.save(buffer, format="PNG")
            image_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
            results.append(image_str)
        return BatchImage(outputs=[{"image": image_str} for image_str in results])


component = AutoScaler(
    DiffusionServer,  # The component to scale
    cloud_compute=L.CloudCompute("gpu-rtx", disk_size=80, preemptible=True),

    # autoscaler args
    min_replicas=1,
    max_replicas=1,
    endpoint="/predict",
    scale_out_interval=0,
    scale_in_interval=600,
    max_batch_size=1,
    timeout_batching=0.3,
    input_type=Text,
    output_type=Image
)

app = L.LightningApp(component)
