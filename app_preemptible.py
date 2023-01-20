# !pip install 'git+https://github.com/Lightning-AI/LAI-API-Access-UI-Component.git'
import lightning as L
from diffusion_with_autoscaler import BatchText, Text, IntervalReplacement, AutoScaler


class MyPythonServer(L.app.components.PythonServer):
    def __init__(self, *args, **kwargs):
        super().__init__(
            input_type=BatchText,
            output_type=BatchText,
            *args,
            **kwargs,
        )
        print(f"Created PythonServer {id(self)}")

    def setup(self):
        print(f"Setting up PythonServer {id(self)}")
        pass

    def predict(self, requests):
        texts = [request.text for request in requests.inputs]
        return BatchText(inputs=[{"text": text} for text in texts])


component = AutoScaler(
    MyPythonServer,
    cloud_compute=L.CloudCompute("gpu-rtx", disk_size=80),
    strategy=IntervalReplacement(interval=7),
    # autoscaler args
    min_replicas=1,
    max_replicas=1,
    endpoint="/predict",
    scale_out_interval=0,
    scale_in_interval=600,
    max_batch_size=6,
    timeout_batching=0.3,
    input_type=Text,
    output_type=Text,
)

app = L.LightningApp(component)
