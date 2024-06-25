"""
This module defines the jobs that are submitted to endpoints in an FL process.

At a high level, this module defines the following jobs:

- `aggregation_job()`: run by Aggregator nodes.
- `local_training_job`: run by Worker nodes.

In addition to these jobs, this module also provides
"""

import typing as t

from flox.process.jobs.aggregation import AggregateJob, DebugAggregateJob
from flox.process.jobs.local_training import DebugLocalTrainJob, LocalTrainJob
from flox.process.jobs.protocols import AggregableJob, TrainableJob, LauncherFunction

# Job: t.TypeAlias = AggregableJob | TrainableJob | LauncherFunction
Job: t.TypeAlias = AggregableJob | TrainableJob
"""
An umbrella typing that encapsulates both ``AggregableJob`` and ``TrainableJob`` protocols
for job impl for both the aggregator and worker nodes (respectively).
"""


__all__ = [
    # Job protocols.
    "Job",
    "AggregableJob",
    "TrainableJob",
    "LauncherFunction",
    # Job impl.
    "AggregateJob",
    "DebugAggregateJob",
    "LocalTrainJob",
    "DebugLocalTrainJob",
]
