import abc
from typing import Any, Optional, Callable
import time
import numpy as np
from fastapi import Request
from lightning import LightningWork, LightningFlow
from lightning.app.structures import List
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import requests
from requests import Response, Session
from lightning.app.structures import Dict
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


class Strategy(abc.ABC, LightningFlow):
    def __init__(self):
        super().__init__()
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
    def run(
        self,
        serve_works: List[LightningWork],
        create_work: Callable,
        replace_work: Callable,
    ) -> Any:
        pass

    def on_after_run(self, serve_works: List[LightningWork], res):
        pass


class PreemptibleRollout(Strategy):
    def __init__(self, interval: int = 30 * 60) -> None:
        """
        This strategy implements a mechanism to automatically replace servers on a scheduled internal
        to continously run on spot instances.

        Arguments:
            interval: Time in seconds before creating a replacement server.

        """
        super().__init__()
        self.interval = interval
        self._work_start_tracker = {}
        self._old_works = []
        self.new_works = List()
        self.fake_trigger = 0

    def run(
        self,
        serve_works: List[LightningWork],
        create_work: Callable,
        replace_work: Callable,
    ) -> None:
        self.fake_trigger += 1  # Note: change state to keep calling `run`.
        # print(self.fake_trigger)
        # serve_works = [w for w in serve_works if w.cloud_compute.preemptible]

        for old_work in serve_works:
            if old_work.url and old_work not in self._work_start_tracker:
                print(f"Tracking preemptive {old_work.name}.")
                self._work_start_tracker[old_work] = time.time()

        for old_work, start_time in self._work_start_tracker.items():
            if self.interval < (time.time() - start_time) and old_work not in self._old_works:
                new_work = create_work()
                print(f"Created a new work {new_work}")
                self.new_works.append(new_work)
                print(f"Appended the new work {new_work} to new_works")
                self._old_works.append(old_work)
                print(f"Appended the old work {old_work} to old_works")

        # spin up machines
        for new_work in self.new_works:
            new_work.run()

        items = zip(self._old_works, self.new_works)
        for old_work, new_work in items:
            print(old_work.status, old_work.url, "->", new_work.status, new_work.url)
            if new_work.url:
                print("calling replace_work(", old_work.url, new_work.url, ")")
                value = replace_work(old_work, new_work)
                print(value)
                # if value is None:
                # wait for the next time
                # elif value:
                #     Worked
                # else:
                #     It didn't work

    def on_after_run(self, serve_works: List[LightningWork], res):
        pass
