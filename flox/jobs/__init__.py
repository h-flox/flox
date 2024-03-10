"""
This module defines the jobs that are submitted to endpoints in an FL process.

At a high level, this module defines the following jobs:

- `aggregation_job()`: run by Aggregator nodes.
- `local_training_job`: run by Worker nodes.

In addition to these jobs, this module also provides 
"""
import typing as t

from flox.jobs.aggregation import AggregateJob, DebugAggregateJob
from flox.jobs.local_training import LocalTrainJob, DebugLocalTrainJob
from flox.jobs.protocols import NodeCallable, AggregableJob, TrainableJob

Job: t.TypeAlias = AggregableJob | TrainableJob | NodeCallable
"""
An umbrella typing that encapsulates both ``AggregableJob`` and ``TrainableJob`` protocols
for job impl for both the aggregator and worker nodes (respectively).
"""


__all__ = [
    # Job protocols.
    "Job",
    "AggregableJob",
    "TrainableJob",
    "NodeCallable",
    # Job impl.
    "AggregateJob",
    "DebugAggregateJob",
    "LocalTrainJob",
    "DebugLocalTrainJob",
]
