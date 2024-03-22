from concurrent.futures import Future

import parsl
from parsl.concurrent import ParslPoolExecutor
from parsl.configs.htex_local import config

from flox.jobs import Job
from flox.runtime.launcher.base import Launcher


class ParslLauncher(Launcher):
    """
    Class that launches tasks via Parsl.
    """

    def __init__(self, resource_spec: dict = None):
        super().__init__()
        parsl.load()

        self.executor = ParslPoolExecutor(config)
        self.spec = {} if resource_spec is None else resource_spec

    def submit(self, job: Job, /, **kwargs) -> Future:
        # print("Trying to submit....")
        # future = self.executor.submit(job, **kwargs)
        # print("Got future!!")
        # return future
        future = parsl.dfk().executors["_parsl_internal"].submit(job, {}, **kwargs)
        return future

    def collect(self):
        raise NotImplementedError
