from flox.jobs.aggregation import AggregateJob, DebugAggregateJob
from flox.jobs.local_training import LocalTrainJob, DebugLocalTrainJob
from flox.jobs.protocols import AggregableJob, TrainableJob


def test_protocol_implementations():
    assert isinstance(AggregateJob(), AggregableJob)
    assert isinstance(DebugAggregateJob(), AggregableJob)
    assert isinstance(LocalTrainJob(), TrainableJob)
    assert isinstance(DebugLocalTrainJob(), TrainableJob)
