# DiffusionWithAutoscaler
`DiffusionWithAutoscaler` allows you to serve stable diffusion with a production ready endpoint on [lightning.ai](https://lightning.ai/).

To get started, save this code snippet as `app.py` and run the below at the end of the README.

### Serve an Autoscaled Stable Diffusion Endpoint

```python
# !pip install 'git+https://github.com/Lightning-AI/stablediffusion.git@lit'
# !pip install 'git+https://github.com/Lightning-AI/DiffusionWithAutoscaler.git'
# !pip install 'git+https://github.com/Lightning-AI/LAI-API-Access-UI-Component.git'
# !curl https://raw.githubusercontent.com/Lightning-AI/stablediffusion/lit/configs/stable-diffusion/v1-inference.yaml -o v1-inference.yaml
import lightning as L
import os, base64, io, ldm, torch
from diffusion_with_autoscaler import AutoScaler, BatchText, BatchImage, Text, Image, CustomColdStartProxy

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
        cmd = "curl -C - https://pl-public-data.s3.amazonaws.com/dream_stable_diffusion/v1-5-pruned-emaonly.ckpt -o v1-5-pruned-emaonly.ckpt"
        os.system(cmd)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model = ldm.lightning.LightningStableDiffusion(
            config_path="v1-inference.yaml",
            checkpoint_path="v1-5-pruned-emaonly.ckpt",
            device=device,
            fp16=True, # Supported on GPU, skipped otherwise.
            use_deepspeed=True, # Supported on Ampere and RTX, skipped otherwise.
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
    cloud_compute=L.CloudCompute("gpu-rtx", disk_size=80),

    # autoscaler args
    min_replicas=1,
    max_replicas=3,
    endpoint="/predict",
    scale_out_interval=0,
    scale_in_interval=600,
    max_batch_size=6,
    timeout_batching=0.3,
    input_type=Text,
    output_type=Image,
    cold_start_proxy=CustomColdStartProxy(proxy_url=PROXY_URL),
)

app = L.LightningApp(component)
```

Run the app for free directly [there](https://lightning.ai/component/UJ7stJI225-Serve%20Dreambooth%20Diffusion).

### Running locally

```bash
lightning run app app.py --setup
```

### Running on cloud

```bash
lightning run app app.py --setup --cloud
```


### Benchmarking

When serving [stable diffusion 1.5](https://github.com/Lightning-AI/stablediffusion) with DDIM 30 steps, you can expect the followings numbers on GPU A10 (gpu-rxt).

| Max Batch Size | Number of users (locust)  | Average (ms)  | Min (ms)  | Max (ms)  |
|---|---|---|---|---|
| 1  | 1  | 2185  | 2124  | 5030  |
| 2  | 2  | 4206  | 2139  | 6418  |
| 4  | 4  | 7524  | 2138  | 10900  |
| 6  | 6  | 10929  | 2135  | 18494  |


To reproduce those numbers, you can do the following:

1. Run the app in the cloud

```bash
lightning run app app.py --setup --cloud
```

2. Wait for the app to be ready for inference in the cloud


3. Launch the load testing app (using locust)

```bash
lightning run app loadtest/app.py --cloud --env SERVER_URL={URL_SERVER}
```

Example:

```bash
lightning run app loadtest/app.py --cloud --env SERVER_URL=https://gcrjp-01gpgyn0kzngryjcap9vpn8aht.litng-ai-03.litng.ai
```

4. From the load testing UI, specify a number of users to be either 1, 2, 4, 6 and launch the load testing

5. The Stable Diffusion model gets faster with time. At convergence, you should observe the same numbers as reported in the table above. Otherwise, open an issue. 

Note: Cuda Graph isn't supported yet, but extra speed up can be expected soon.