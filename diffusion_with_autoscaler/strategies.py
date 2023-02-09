import abc
from typing import Any, Optional, Callable
import time
from lightning import LightningWork, LightningFlow
from lightning.app.structures import List


class Strategy(abc.ABC, LightningFlow):

    @abc.abstractmethod
    def run(
        self,
        serve_works: List[LightningWork],
        create_work: Callable,
        register_work: Callable,
        replace_work: Callable,
    ) -> Any:
        """Controls the following:
        - when to create works
        - how to create works, e.g. create_work(weight_path="path/to/new_weights")
        - when to replace old works with new works
        """
        pass

    def on_after_run(self, serve_works: List[LightningWork], res):
        pass


class IntervalReplacement(Strategy):
    def __init__(self, interval: int = 30 * 60) -> None:
        """
        This strategy implements a mechanism to automatically replace servers on a scheduled interval
        to continously run on spot instances.

        Arguments:
            interval: Time in seconds before creating a replacement server.
        """
        super().__init__()
        self.interval = interval
        self._work_start_tracker = {}  # {w1: t1_since_tracked, w2: t2_since_tracked, ...}
        self._old_to_new_work = {}  # {old_w1: new_w1, old_w2: new_w2, ...}

    def run(
        self,
        serve_works: List[LightningWork],
        create_work: Callable,
        register_work: Callable,
        replace_work: Callable,
    ) -> None:
        # step 1: collect running works to replace
        for work in serve_works:
            if work.url and work not in self._work_start_tracker:
                print(f"Started tracking old work {work.name}")
                self._work_start_tracker[work] = time.time()

        # step 2: ask autoscaler to launch new works to replace old works with later
        for old_work, start_time in self._work_start_tracker.items():
            if (time.time() - start_time) < self.interval:
                continue

            if old_work not in self._old_to_new_work:
                new_work = create_work()
                print(f"Created a new work {new_work.name}")
                _ = register_work(old_work, new_work)  # autoscaler will launch new_work in the background
                print(f"Registered new work {new_work.name}")
                self._old_to_new_work[old_work] = new_work  # holds which old work to replace with the new work
                self._work_start_tracker[old_work] = time.time()
            else:
                print(
                    f"Skipped creating a new work as already created {self._old_to_new_work[old_work].name}"
                    f" for the old work {old_work.name}"
                )

        # step 3: replace old works with new works if new ones are ready
        for old_work, new_work in {**self._old_to_new_work}.items():
            if new_work.url:
                replace_work(old_work, new_work)
                print(f"Replaced with {old_work} with {new_work}")
                del self._work_start_tracker[old_work]
                del self._old_to_new_work[old_work]
