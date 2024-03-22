import typing as t
from concurrent.futures import Future

import parsl
from parsl.executors import HighThroughputExecutor

from flox.jobs import Job
from flox.runtime.launcher.base import Launcher


class ParslLauncher(Launcher):
    """
    Class that launches tasks via Parsl.
    """

    def __init__(self, config: dict[str, t.Any]):
        super().__init__()
        parsl.load()

        # self.executor = ParslPoolExecutor(config)
        self._config = config
        self.executor = HighThroughputExecutor(**self._config)
        self.executor.run_dir = "."
        self.executor.provider.script_dir = "."
        self.executor.start()
        # self.spec = {} if resource_spec is None else resource_spec

    def submit(self, job: Job, /, **kwargs) -> Future:
        # print("Trying to submit....")
        # future = self.executor.submit(job, **kwargs)
        # print("Got future!!")
        # return future

        # future = parsl.dfk().executors["_parsl_internal"].submit(job, {}, **kwargs)
        # return future

        future = self.executor.submit(job, {}, **kwargs)
        return future

    def collect(self):
        raise NotImplementedError
