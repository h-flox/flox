from typing import TypeAlias

from proxystore.proxy import Proxy

from flox.reporting.job import JobResult

Result: TypeAlias = Proxy[JobResult] | JobResult