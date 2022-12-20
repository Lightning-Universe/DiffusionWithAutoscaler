# !pip install lightning_api_access
# !pip install 'git+https://github.com/Lightning-AI/stablediffusion.git@lit'
# !curl https://raw.githubusercontent.com/Lightning-AI/stablediffusion/main/configs/stable-diffusion/v2-inference-v.yaml -o v2-inference-v.yaml
import lightning as L
import os
from time import sleep

from autoscaler import AutoScaler
from datatypes import BatchText, BatchResponse, Text, Image

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"


class EmulatorDiffusionServer(L.app.components.PythonServer):
    def __init__(self, *args, **kwargs):
        super().__init__(
            input_type=BatchText,
            output_type=BatchResponse,
            *args,
            **kwargs,
        )

    def setup(self):
        pass
        # sleep(0)

    def predict(self, requests):
        batch_size = len(requests.inputs)

        print(batch_size)
        
        sleep_times = {
            1: 10,
            2: 16,
            3: 24,
            4: 30,
            5: 36,
            6: 46,
            7: 53,
            8: 59,
        }
        sleep(sleep_times[batch_size])
        return BatchResponse(outputs=[{"image": "image_str"} for _ in range(batch_size)])


component = AutoScaler(
    EmulatorDiffusionServer,  # The component to scale
    cloud_compute=L.CloudCompute("gpu-rtx", disk_size=80),

    # autoscaler args
    min_replicas=1,
    max_replicas=1,
    endpoint="/predict",
    scale_out_interval=0,
    scale_in_interval=30,
    max_batch_size=6,
    timeout_batching=4,
    input_type=Text,
    output_type=Image,
    # cold_start_proxy=CustomColdStartProxy(),
)

app = L.LightningApp(component)
