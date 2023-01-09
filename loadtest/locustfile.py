import os

from locust import FastHttpUser, task

TEXT = "astronaut riding a horse, digital art, epic lighting, highly-detailed masterpiece trending HQ"
SERVER_URL = os.getenv("SERVER_URL", None)


class User(FastHttpUser):
    if SERVER_URL:
        host = SERVER_URL

    @task
    def req(self):
        self.client.post("/predict", json={"text": TEXT})
