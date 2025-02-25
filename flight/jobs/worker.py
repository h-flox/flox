class WorkerState:
    pass


def worker_job():
    from flight.events import WorkerEvents
    from flight.learning.module import TorchModule, TorchDataModule

    from ignite.engine import (
        create_supervised_trainer,
        Engine,
    )
    from torch.utils.data import DataLoader, Dataset

    ####################################################################

    WorkerEvents.STARTED

    ####################################################################

    ...  # TODO

    ####################################################################

    WorkerEvents.COMPLETED

    return None
