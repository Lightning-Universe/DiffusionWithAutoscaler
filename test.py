import base64
from pathlib import Path
import requests

response = requests.post('https://uxiej-01gqfmfkev243487nznybjfbz2.litng-ai-03.litng.ai/predict', json={
"text": "A portrait of a person looking away from the camera"
})
# If you are using basic authentication for your app, you should add your credentials to the request:
# response = requests.post('https://uxiej-01gqfmfkev243487nznybjfbz2.litng-ai-03.litng.ai/predict', json={
# "text": "A portrait of a person looking away from the camera"
# }, auth=requests.auth.HTTPBasicAuth('your_username', 'your_password'))

img = response.json()["image"]
img = base64.b64decode(img.encode("utf-8"))
Path("response.png").write_bytes(img)