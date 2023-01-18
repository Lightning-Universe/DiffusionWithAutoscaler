import base64
from typing import Optional, Dict, Any, List

import requests
from pydantic import BaseModel
from textwrap import dedent


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
            """import base64, requests
from pathlib import Path

imgurl = "https://raw.githubusercontent.com/Lightning-AI/LAI-Triton-Server-Component/main/catimage.png"
img = requests.get(imgurl).content
img = base64.b64encode(img).decode("UTF-8")
response = requests.post('"""
            + url
            + """', json={
"image": img
})
# If you are using basic authentication for your app, you should add your credentials to the request:
# response = requests.post('"""
            + url
            + """', json={
# "image": img
# }, auth=requests.auth.HTTPBasicAuth('your_username', 'your_password'))
"""
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
# If you are using basic authentication for your app, you should add your credentials to the request:
# response = requests.post('"""
            + url
            + """', json={
# "text": "A portrait of a person looking away from the camera"
# }, auth=requests.auth.HTTPBasicAuth('your_username', 'your_password'))
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
# If you are using basic authentication for your app, you should add your credentials to the request:
# response = requests.post('"""
            + url
            + """', json={
# "inputs": [{"text": "A portrait of a person looking away from the camera"}],
# }, auth=requests.auth.HTTPBasicAuth('your_username', 'your_password'))
"""
        )


class BatchImage(BaseModel):
    # Note: field name must be `outputs`
    outputs: List[Image]

    @staticmethod
    def response_code_sample() -> str:
        return """img = response.json()["outputs"][0]["image"]
img = base64.b64decode(img.encode("utf-8"))
Path("response.png").write_bytes(img)
"""


class TextImage(BaseModel):
    text: Optional[str]
    image: Optional[str]

    @staticmethod
    def get_sample_data() -> Dict[Any, Any]:
        return {"text": "A fantasy landscape, trending on artstation"}

    @staticmethod
    def request_code_sample(url: str) -> str:
        return dedent(
            """import base64
from pathlib import Path
import requests

url = "https://raw.githubusercontent.com/CompVis/stable-diffusion/main/assets/stable-samples/img2img/sketch-mountains-input.jpg"
image = requests.get(url).content
image = base64.b64encode(image).decode("ascii")

response = requests.post('"""
            + url
            + """', 
    json={
        "text": "A fantasy landscape, trending on artstation", 
        "image": image
    }
)

# If you are using basic authentication for your app, you should add your credentials to the request:
# response = requests.post('"""
            + url
            + """',
#    json={
#        "text": "A fantasy landscape, trending on artstation", 
#        "image": image,
#    },
#    auth=requests.auth.HTTPBasicAuth('your_username', 'your_password')
#)"""
        )


class BatchTextImage(BaseModel):
    # Note: field name must be `inputs`
    inputs: List[TextImage]

    @staticmethod
    def request_code_sample(url: str) -> str:
        return dedent(
            """import base64
from pathlib import Path
import requests

url = "https://raw.githubusercontent.com/CompVis/stable-diffusion/main/assets/stable-samples/img2img/sketch-mountains-input.jpg"
image = requests.get(url).content
image = base64.b64encode(image).decode("ascii")

response = requests.post('"""
            + url
            + """', 
    json={
        "inputs": [{
            "text": "A fantasy landscape, trending on artstation", 
            "image": image,
        }]
    }
)

# If you are using basic authentication for your app, you should add your credentials to the request:
# response = requests.post('"""
            + url
            + """',
#    json={
#        "inputs": [{
#            "text": "A fantasy landscape, trending on artstation", 
#            "image": image,
#        }]
#    },
#    auth=requests.auth.HTTPBasicAuth('your_username', 'your_password')
#)"""
        )
