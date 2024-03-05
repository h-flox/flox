from __future__ import annotations

import typing as t

from flox.jobs.protocols import TrainableJob

if t.TYPE_CHECKING:
    from flox.data import FloxDataset
    from flox.flock import FlockNode
    from flox.nn import FloxModule
    from flox.nn.typing import StateDict
    from flox.runtime import Result
    from flox.runtime.transfer import BaseTransfer
    from flox.strategies import WorkerStrategy, TrainerStrategy


class LocalTrainJob(TrainableJob):
    @staticmethod
    def __call__(
        node: FlockNode,
        parent: FlockNode,
        module: FloxModule,
        module_state_dict: StateDict,
        dataset: FloxDataset,
        transfer: BaseTransfer,
        worker_strategy: WorkerStrategy,
        trainer_strategy: TrainerStrategy,
        **train_hyper_params,
    ) -> Result:
        """Perform local training on a worker node.

        Args:
            node (FlockNode):
            transfer (BaseTransfer): ...
            parent (FlockNode):
            strategy (Strategy):
            module (FloxModule):
            module_state_dict (StateDict):
            dataset (Dataset | Subset | None):
            **train_hyper_params ():

        Returns:
            Local fitting results.
        """
        from copy import deepcopy
        from flox.flock.states import WorkerState
        from flox.nn.trainer import Trainer
        from torch.utils.data import DataLoader
        from flox.runtime import JobResult

        global_model = module
        global_state_dict = module.state_dict()
        local_model = deepcopy(module)
        global_model.load_state_dict(module_state_dict)
        local_model.load_state_dict(module_state_dict)

        node_state = WorkerState(
            node.idx,
            pre_local_train_model=global_model,
            post_local_train_model=local_model,
        )

        worker_strategy.work_start()
        data = dataset.load(node)
        train_dataloader = DataLoader(
            data,
            batch_size=train_hyper_params.get("batch_size", 32),
            shuffle=train_hyper_params.get("shuffle", True),
        )

        # Add optimizer to this strategy.
        worker_strategy.before_training(node_state, data)
        trainer = Trainer()
        optimizer = local_model.configure_optimizer()
        history = trainer.fit(
            local_model,
            optimizer,
            train_dataloader,
            # TODO: Include `trainer_params` as an argument to
            #       this so users can easily customize Trainer.
            num_epochs=train_hyper_params.get("num_epochs", 2),
            node_state=node_state,
            trainer_strategy=trainer_strategy,
        )

        local_params = worker_strategy.after_training(node_state)

        history["node/idx"] = node.idx
        history["node/kind"] = node.kind.to_str()
        history["parent/idx"] = parent.idx
        history["parent/kind"] = parent.kind.to_str()

        result = JobResult(node_state, node.idx, node.kind, local_params, history)
        return transfer.report(result)


class DebugLocalTrainJob(TrainableJob):
    @staticmethod
    def __call__(
        node: FlockNode,
        parent: FlockNode,
        module: FloxModule,
        module_state_dict: StateDict,
        dataset: FloxDataset,
        transfer: BaseTransfer,
        worker_strategy: WorkerStrategy,
        trainer_strategy: TrainerStrategy,
        **train_hyper_params,
    ):  # -> Result:
        """

        Args:
            node ():
            transfer ():
            parent ():
            strategy ():
            module (FloxModule): ...

        Returns:

        """
        import datetime
        import numpy as np
        import pandas
        from flox.flock.states import WorkerState
        from flox.runtime import JobResult

        local_module = module
        node_state = WorkerState(
            node.idx,
            pre_local_train_model=local_module,
            post_local_train_model=local_module,
        )
        history = {
            "node/idx": [node.idx],
            "node/kind": [node.kind.to_str()],
            "parent/idx": [parent.idx],
            "parent/kind": [parent.kind.to_str()],
            "train/loss": [np.nan],
            "train/epoch": [np.nan],
            "train/batch_idx": [np.nan],
            "train/time": [datetime.datetime.now()],
            "mode": "debug",
        }
        history_df = pandas.DataFrame.from_dict(history)
        result = JobResult(
            node_state, node.idx, node.kind, module.state_dict(), history_df
        )
        return transfer.report(result)
