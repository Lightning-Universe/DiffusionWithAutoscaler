# !pip install 'git+https://github.com/Lightning-AI/stablediffusion.git@lit'
# !pip install 'git+https://github.com/Lightning-AI/LAI-API-Access-UI-Component.git'
# !curl https://raw.githubusercontent.com/Lightning-AI/stablediffusion/lit/configs/stable-diffusion/v1-inference.yaml -o v1-inference.yaml
import lightning as L
import asyncio
import uuid
import torch
from diffusion_with_autoscaler import BatchText, BatchImage, Image

PROXY_URL = "https://ulhcn-01gd3c9epmk5xj2y9a9jrrvgt8.litng-ai-03.litng.ai/api/predict"

class DiffusionServer(L.app.components.PythonServer):
    def __init__(self, *args, **kwargs):
        super().__init__(
            input_type=BatchText,
            output_type=BatchImage,
            port=5050,
            *args,
            **kwargs,
        )

        self._requests = {}
        self._predictor_task = None
        self._lock = None
        self._tokenizer = None
    
    def setup(self):
        self._tokenizer = lambda x: x if isinstance(x, torch.Tensor) else torch.tensor(1)

    def inference(self, requests):
        inputs = [self._tokenizer(request['data']) for request in requests]
        print(inputs)
        if len(inputs) == 1:
            inputs[0] += 1
        else:
            inputs = torch.stack(inputs)
            inputs += 1
            inputs = torch.unbind(inputs)
        for request, input in zip(requests, inputs):
            request['data'] = input
        return [None for _ in requests]

    async def predict_fn(self):
        while True:
            async with self._lock:
                keys = list(self._requests)

            if len(keys) == 0:
                await asyncio.sleep(0.05)
                continue
            
            requests = [self._requests[key] for key in keys]
            results = self.inference(requests)

            for key, result in zip(keys, results):
                if result:
                    self._requests[key]['response'].set_result(result)
                    del self._requests[key]

            await asyncio.sleep(0.05)

    async def predict(self, request: BatchText):
        if self._lock is None:
            self._lock = asyncio.Lock()
        if self._predictor_task is None:
            self._predictor_task = asyncio.create_task(self.predict_fn())
        assert len(request.inputs) == 1
        future = asyncio.Future()
        async with self._lock:
            self._requests[uuid.uuid4().hex] = {"data": request.inputs[0], "response": future}
        result = await future
        print('Out', result)
        return result


# component = AutoScaler(
#     DiffusionServer,  # The component to scale
#     cloud_compute=L.CloudCompute("gpu-rtx", disk_size=80),

#     # autoscaler args
#     min_replicas=1,
#     max_replicas=1,
#     endpoint="/predict",
#     scale_out_interval=0,
#     scale_in_interval=600,
#     max_batch_size=6,
#     timeout_batching=0.3,
#     input_type=Text,
#     output_type=Image,
#     batching="continuous",
# )
component = DiffusionServer()
app = L.LightningApp(component)
