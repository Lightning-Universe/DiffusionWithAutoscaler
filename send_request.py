# import time
# import base64
# from pathlib import Path
# import requests

# response = requests.post('https://psqsk-01gpx940tvszz8apsxgr43xzs2.litng-ai-03.litng.ai/predict', json={
# "text": "A portrait of a person looking away from the camera"
# })
# # If you are using basic authentication for your app, you should add your credentials to the request:
# # response = requests.post('https://psqsk-01gpx940tvszz8apsxgr43xzs2.litng-ai-03.litng.ai/predict', json={
# # "text": "A portrait of a person looking away from the camera"
# # }, auth=requests.auth.HTTPBasicAuth('your_username', 'your_password'))

# img = response.json()["image"]
# img = base64.b64decode(img.encode("utf-8"))
# Path(f"response_{time.time()}.png").write_bytes(img)


import base64
from pathlib import Path
import requests

response = requests.post('http://localhost:50151/predict', json={"text": "A portrait of a person looking away from the camera"})

print(response, response.json())

img = response.json()["image"]
img = base64.b64decode(img.encode("utf-8"))
Path("response.png").write_bytes(img)
