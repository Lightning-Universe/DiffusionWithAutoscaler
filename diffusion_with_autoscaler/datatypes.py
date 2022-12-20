import base64
from typing import Optional, Dict, Any, List

import requests
from pydantic import BaseModel


class Image(BaseModel):
    image: Optional[str]

    @staticmethod
    def get_sample_data() -> Dict[Any, Any]:
        url = "https://raw.githubusercontent.com/Lightning-AI/LAI-Triton-Server-Component/main/catimage.png"
        img = requests.get(url).content
        img = base64.b64encode(img).decode("UTF-8")
        return {"image": img}

    @staticmethod
    def request_code_sample(url: str) -> str:
        return (
                """import base64
from pathlib import Path
import requests

imgurl = "https://raw.githubusercontent.com/Lightning-AI/LAI-Triton-Server-Component/main/catimage.png"
img = requests.get(imgurl).content
img = base64.b64encode(img).decode("UTF-8")
response = requests.post('"""
            + url
            + """', json={
"image": img
})"""
        )

    @staticmethod
    def response_code_sample() -> str:
        return """img = response.json()["image"]
img = base64.b64decode(img.encode("utf-8"))
Path("response.png").write_bytes(img)
"""


class Text(BaseModel):
    text: Optional[str]

    @staticmethod
    def get_sample_data() -> Dict[Any, Any]:
        return {"text": "A portrait of a person looking away from the camera"}

    @staticmethod
    def request_code_sample(url: str) -> str:
        return (
                """import base64
from pathlib import Path
import requests

response = requests.post('"""
            + url
            + """', json={
"text": "A portrait of a person looking away from the camera"
})
"""
        )


class BatchText(BaseModel):
    # Note: field name must be `inputs`
    inputs: List[Text]

    @staticmethod
    def request_code_sample(url: str) -> str:
        return (
                """import base64
from pathlib import Path
import requests
response = requests.post('"""
                + url
                + """', json={
"inputs": [{"text": "A portrait of a person looking away from the camera"}]
})
"""
        )


class BatchResponse(BaseModel):
    # Note: field name must be `outputs`
    outputs: List[Image]

    @staticmethod
    def response_code_sample() -> str:
        return """img = response.json()["outputs"][0]["image"]
img = base64.b64decode(img.encode("utf-8"))
Path("response.png").write_bytes(img)
"""
