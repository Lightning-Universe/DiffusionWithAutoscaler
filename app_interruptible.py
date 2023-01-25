# !pip install 'git+https://github.com/Lightning-AI/LAI-API-Access-UI-Component.git'
import lightning as L
from typing import List
from pydantic import BaseModel
from diffusion_with_autoscaler import BatchText, Text, IntervalReplacement, AutoScaler


class BatchTextInput(BaseModel):
    # Note: field name must be `inputs`
    inputs: List[Text]


class BatchTextOutput(BaseModel):
    # Note: field name must be `outputs`
    outputs: List[Text]


class MyPythonServer(L.app.components.PythonServer):
    def __init__(self, *args, **kwargs):
        super().__init__(
            input_type=BatchTextInput,
            output_type=BatchTextOutput,
            *args,
            **kwargs,
        )
        print(f"Created PythonServer {id(self)}")

    def setup(self):
        print(f"Setting up PythonServer {id(self)}")
        pass

    def predict(self, requests: BatchTextInput):
        print(f"predicting on {self.url}:{self.port}")  # check inference done on a new work
        texts = [request.text for request in requests.inputs]
        return BatchTextOutput(outputs=[{"text": text} for text in texts])


component = AutoScaler(
    MyPythonServer,
    # cloud_compute=L.CloudCompute("gpu-rtx", disk_size=80, preemptible=True),
    # strategy=IntervalReplacement(interval=5*60),  # replace every 5 minutes
    cloud_compute=L.CloudCompute("cpu-medium", disk_size=80),  # for debugging
    strategy=IntervalReplacement(interval=10),
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
