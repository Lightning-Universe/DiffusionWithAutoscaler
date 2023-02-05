import os
import time

from locust import FastHttpUser, task

TEXT = "A cat running away from a mouse"
SERVER_URL = os.getenv("SERVER_URL", None)


class User(FastHttpUser):
    if SERVER_URL:
        host = SERVER_URL

    @task
    def req(self):
        self.client.post("/predict", json={"text": TEXT})
