"""
This module defines the jobs that are submitted to endpoints in an FL federation.

At a high level, this module defines the following jobs:

- `aggregation_job()`: run by Aggregator nodes.
- `local_training_job`: run by Worker nodes.

In addition to these jobs, this module also provides
"""

import typing as t

from flox.federation.jobs.aggr import AggregateJob, DebugAggregateJob
from flox.federation.jobs.protocols import AggrJob, LauncherFunction, TrainJob
from flox.federation.jobs.train import DebugLocalTrainJob, LocalTrainJob

# Job: t.TypeAlias = AggregableJob | TrainableJob | LauncherFunction
Job: t.TypeAlias = AggrJob | TrainJob
"""
An umbrella typing that encapsulates both ``AggregableJob`` and ``TrainableJob`` protocols
for job impl for both the aggregator and worker nodes (respectively).
"""


__all__ = [
    # Job protocols.
    "Job",
    "AggrJob",
    "TrainJob",
    "LauncherFunction",
    # Job impl.
    "AggregateJob",
    "DebugAggregateJob",
    "LocalTrainJob",
    "DebugLocalTrainJob",
]
