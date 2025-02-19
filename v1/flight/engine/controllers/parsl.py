from __future__ import annotations

import typing as t
from concurrent.futures import Future

from .base import AbstractController

if t.TYPE_CHECKING:
    import pathlib

    from parsl.executors import HighThroughputExecutor

    from v1.flight.types import P, T


def _platform_info() -> str:
    import platform

    return platform.platform()


class ParslController(AbstractController):
    """
    Controller implementation for using the **Parsl** workflow manager.

    This class is especially useful for high-throughput simulations of Federated
    Learning on a *high-performance computing* (HPC) system.

    More information about how Parsl can be used to parallelize workflows across
    HPC systems can be found
    [here](https://parsl.readthedocs.io/en/stable/userguide/configuring.html).
    """

    _config: t.Any
    _executor: HighThroughputExecutor

    def __init__(
        self,
        config: dict[str, t.Any],
        run_dir: pathlib.Path | str,
        script_dir: pathlib.Path | str,
        priming: bool = True,
    ):
        """
        Args:
            config (dict[str, t.Any]):
            run_dir (pathlib.Path | str):
            script_dir (pathlib.Path | str):
            priming (bool): If `True`, a simple priming function is run to reduce
                initial startup costs during execution of aggr. This priming job just
                simply returns platform information. No priming is done if `False`.
        """
        import parsl
        from parsl.executors import HighThroughputExecutor

        parsl.load()

        self._config = config
        self._executor = HighThroughputExecutor(**self._config)
        self._executor.run_dir = run_dir if isinstance(run_dir, str) else str(run_dir)
        self._executor.provider.script_dir = script_dir
        self._executor.start()

        if priming:
            fut = self.submit(_platform_info)
            fut.result()

    def submit(self, fn: t.Callable[P, T], /, **kwargs) -> Future[T]:  # noqa
        future = self._executor.submit(fn, {}, **kwargs)
        return future

    def shutdown(self):
        """
        Shuts down the Parsl `HighThroughputExecutor`.
        """
        self._executor.shutdown()
