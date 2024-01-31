"""
This module defines the jobs that are submitted to endpoints in an FL process.

At a high level, this module defines the following jobs:
- `aggregation_job()`: run by Aggregator nodes.
- `local_training_job`: run by Worker nodes.

In addition to these jobs, this module also provides 
"""

from flox.runtime.jobs.aggr import aggregation_job
from flox.runtime.jobs.train import local_training_job, debug_training_job

__all__ = ["aggregation_job", "local_training_job", "debug_training_job"]
