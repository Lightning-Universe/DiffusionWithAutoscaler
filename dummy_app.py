import lightning as L
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
        pass

    def predict(self, requests):
        texts = [request.text for request in requests.inputs]
        return BatchImage(outputs=[{"image": "NONE"} for _ in texts])


component = AutoScaler(
    DiffusionServer,  # The component to scale
    cloud_compute=L.CloudCompute("gpu-rtx", disk_size=80, preemptible=True),

    strategy=PreemptibleRollout(interval=5),
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