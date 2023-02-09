# !pip install 'git+https://github.com/Lightning-AI/LAI-API-Access-UI-Component.git'
import lightning as L
import time
from diffusion_with_autoscaler import AutoScaler, BatchText, BatchImage, Text, Image, IntervalReplacement


class DiffusionServer(L.app.components.PythonServer):
    def __init__(self, *args, **kwargs):
        super().__init__(
            input_type=BatchText,
            output_type=BatchImage,
            *args,
            **kwargs,
        )

    def setup(self):
        print(f"{time.time()}: {self.__class__}.setup()")

    def predict(self, requests):
        print(f"{time.time()}: {self.__class__}.predict()")
        texts = [request.text for request in requests.inputs]
        results = texts
        time.sleep(6 * len(requests.inputs))
        return BatchImage(outputs=[{"image": image_str} for image_str in results])


component = AutoScaler(
    DiffusionServer,  # The component to scale
    cloud_compute=L.CloudCompute("gpu", interruptible=True),
    strategy=IntervalReplacement(interval=10),  # in seconds
    # autoscaler args
    min_replicas=1,
    max_replicas=8,
    endpoint="/predict",
    scale_out_interval=0,  # scale out ASAP
    scale_in_interval=60 * 10,  # scale in every 600 seconds
    max_batch_size=6,
    timeout_batching=0.3,
    input_type=Text,
    output_type=Image,
)

app = L.LightningApp(component)
