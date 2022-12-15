from typing import Any

import aiohttp
from pydantic import BaseModel

from datatypes import Image, Text

proxy_url = "https://ulhcn-01gd3c9epmk5xj2y9a9jrrvgt8.litng-ai-03.litng.ai/api/predict"


class ColdStartProxy:
    def __init__(self, proxy_url):
        self.proxy_url = proxy_url
        self.proxy_timeout = 20

    async def handle_request(self, request: BaseModel) -> Any:
        # TODO default handler
        pass


class CustomColdStartProxy(ColdStartProxy):

    def __init__(self):
        super().__init__(proxy_url)

    async def handle_request(self, request: Text) -> Any:
        # TODO - batch processing in the cold start proxy
        try:
            async with aiohttp.ClientSession() as session:
                headers = {
                    "accept": "application/json",
                    "Content-Type": "application/json",
                }
                async with session.post(
                        self.proxy_url,
                        json={"prompt": request.text},
                        timeout=self.proxy_timeout,
                        headers=headers,
                ) as response:
                    resp = await response.json()
                    return Image(image=resp["image"][22:])
        except Exception as ex:
            # TODO - exception raising
            print(f"Error in proxy: {ex}")
            return None
