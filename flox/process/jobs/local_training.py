from __future__ import annotations

import typing as t

from flox.process.jobs.protocols import TrainableJob

if t.TYPE_CHECKING:
    from flox.learn import FloxModule
    from flox.learn.data import FloxDataset
    from flox.learn.types import Params
    from flox.runtime import Result
    from flox.runtime.transfer import TransferProtocol
    from flox.strategies import TrainerStrategy, WorkerStrategy
    from flox.topos import Node


class LocalTrainJob(TrainableJob):
    @staticmethod
    def __call__(
        node: Node,
        parent: Node,
        global_model: FloxModule,
        module_state_dict: Params,
        dataset: FloxDataset,
        transfer: TransferProtocol,
        worker_strategy: WorkerStrategy,
        trainer_strategy: TrainerStrategy,
        **train_hyper_params,
    ) -> Result:
        """Perform local training on a worker node.

        Args:
            node (Node):
            transfer (TransferProtocol): ...
            parent (Node):
            strategy (Strategy):
            module (FloxModule):
            module_state_dict (Params):
            dataset (Dataset | Subset | None):
            **train_hyper_params ():

        Returns:
            Local fitting results.
        """

        from copy import deepcopy
        from datetime import datetime

        from proxystore.proxy import Proxy, extract
        from torch.utils.data import DataLoader

        from flox.learn.trainer import Trainer
        from flox.runtime import JobResult
        from flox.topos import WorkerState

        training_start = datetime.now()
        local_model = deepcopy(global_model)

        if isinstance(module_state_dict, Proxy):
            module_state_dict = extract(module_state_dict)

        global_model.load_state_dict(module_state_dict)
        local_model.load_state_dict(module_state_dict)
        state = WorkerState(
            node.idx,
            global_model=global_model,
            local_model=local_model,
        )

        state = worker_strategy.work_start(state)
        data = dataset.load(node)
        train_dataloader = DataLoader(
            data,
            batch_size=train_hyper_params.get("batch_size", 32),
            shuffle=train_hyper_params.get("shuffle", True),
        )

        optimizer = local_model.configure_optimizers()
        state, data = worker_strategy.before_training(state, data)
        trainer = Trainer(trainer_strategy)
        history = trainer.fit(
            local_model,
            optimizer,
            train_dataloader,
            num_epochs=train_hyper_params.get("num_epochs", 5),
            node_state=state,
        )

        state = worker_strategy.after_training(state, optimizer)

        ################################################################################
        # TRAINING DATA POST-PROCESSING
        ################################################################################
        history["training_start"] = training_start
        history["training_end"] = datetime.now()
        history["node/idx"] = node.idx
        history["node/kind"] = node.kind.to_str()
        history["parent/idx"] = parent.idx
        history["parent/kind"] = parent.kind.to_str()

        assert state.local_model is not None
        local_params = state.local_model.state_dict()
        result = JobResult(state, node.idx, node.kind, local_params, history)

        result = worker_strategy.work_end(result)  # NOTE: Double-check.
        return transfer.transfer(result)

    @property
    def __name__(self) -> str:
        return "LocalTrainJob"


class DebugLocalTrainJob(TrainableJob):
    @staticmethod
    def __call__(
        node: Node,
        parent: Node,
        global_model: FloxModule,
        module_state_dict: Params,
        dataset: FloxDataset,
        transfer: TransferProtocol,
        worker_strategy: WorkerStrategy,
        trainer_strategy: TrainerStrategy,
        **train_hyper_params,
    ) -> Result:
        """

        Args:
            node ():
            transfer ():
            parent ():
            strategy ():
            module (FloxModule): ...

        Returns:

        """

        from datetime import datetime

        import numpy as np
        import pandas

        from flox.runtime import JobResult
        from flox.topos import WorkerState

        local_module = global_model
        node_state = WorkerState(
            node.idx,
            global_model=local_module,
            local_model=local_module,
        )
        history = {
            "node/idx": [node.idx],
            "node/kind": [node.kind.to_str()],
            "parent/idx": [parent.idx],
            "parent/kind": [parent.kind.to_str()],
            "train/loss": [np.nan],
            "train/epoch": [np.nan],
            "train/batch_idx": [np.nan],
            "train/time": [datetime.now()],
            "mode": "debug",
        }
        history_df = pandas.DataFrame.from_dict(history)
        result = JobResult(
            node_state, node.idx, node.kind, global_model.state_dict(), history_df
        )
        return transfer.transfer(result)

    @property
    def __name__(self) -> str:
        return "DebugLocalTrainJob"
