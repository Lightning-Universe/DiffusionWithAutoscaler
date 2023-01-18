import abc
from typing import Any, Optional, Callable
import time
import numpy as np
from fastapi import Request
from lightning import LightningWork
from lightning.app.structures import List
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import requests
from requests import Response, Session
from lightning import LightningWork
from lightning.app.structures import List
from lightning.app.utilities.exceptions import CacheMissException

_CONNECTION_RETRY_TOTAL = 5
_CONNECTION_RETRY_BACKOFF_FACTOR = 0.5


def get_url(work: LightningWork) -> Optional[str]:
    internal_ip = work.internal_ip
    if internal_ip:
        return f"http://{internal_ip}:{work.port}"
    raise CacheMissException


def _configure_session() -> Session:
    """Configures the session for GET and POST requests.
    It enables a generous retrial strategy that waits for the application server to connect.
    """
    retry_strategy = Retry(
        # wait time between retries increases exponentially according to: backoff_factor * (2 ** (retry - 1))
        total=_CONNECTION_RETRY_TOTAL,
        backoff_factor=_CONNECTION_RETRY_BACKOFF_FACTOR,
        status_forcelist=[429, 500, 502, 503, 504],
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    http = requests.Session()
    http.mount("https://", adapter)
    http.mount("http://", adapter)
    return http


class Strategy(abc.ABC):
    def __init__(self):
        self._session = None

    def select_url(self, request, local_router_metadata):
        method = request.method.lower()
        keys = list(local_router_metadata)
        if len(keys) > 1:
            selected_url = np.random.choice(keys, p=list(local_router_metadata.values()))
        else:
            selected_url = keys[0]
        return selected_url, method

    def make_request(self, request: Request, full_path: str, local_router_metadata: Any, payload) -> Response:

        if self._session is None:
            self._session = _configure_session()

        selected_url, method = self.select_url(request, local_router_metadata)
        if method == "post":
            return getattr(self._session, method)(selected_url + "/" + full_path, json=payload)
        else:
            return getattr(self._session, method)(selected_url + "/" + full_path)

    @abc.abstractmethod
    def run(self, serve_works: List[LightningWork], add_work: Callable) -> Any:
        pass

    def on_after_run(self, serve_works: List[LightningWork], res):
        pass


class PreemptibleRollout(Strategy):
    def __init__(
        self,
        interval: int = 30 * 60,
        timeout: int = 10 * 60,
    ) -> None:
        super().__init__()
        self.interval = interval
        self.timeout = timeout
        self.last_run_time = None
        self.has_run = False
        self.work_start_tracker = {}
        self.replacement_works = {}

    def run(self, serve_works: List[LightningWork], add_work: Callable) -> None:
        replacement_works = list(self.replacement_works)
        for new_work in replacement_works:
            if new_work.url:
                previous_work = self.replacement_works.pop()
                previous_work.stop()

        for work in serve_works:
            if work.url and work.name not in self.work_start_tracker:
                self.work_start_tracker[work] = time.time()

        for work, start_time in self.work_start_tracker.items():
            if self.interval < (time.time() - start_time):
                new_work = work.clone()
                add_work(new_work)
                self.replacement_works[new_work] = work


    def on_after_run(self, serve_works: List[LightningWork], res):
        pass
