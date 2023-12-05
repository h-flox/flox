from typing import TypeAlias

from proxystore.proxy import Proxy

from flox.reporting.job import JobResult

Result: TypeAlias = JobResult | Proxy[JobResult]

__all__ = ["Result", "JobResult"]
