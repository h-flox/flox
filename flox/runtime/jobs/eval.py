def local_training_job(
    node: FlockNode,
    transfer: BaseTransfer,
    parent: FlockNode,
    strategy: Strategy,
    module: FloxModule,
    module_state_dict: StateDict,
    dataset: Dataset | Subset | None = None,  # TODO: Cannot be `None`.
    **train_hyper_params,
) -> Result:
    pass


def aggr_training_job(
    node: FlockNode,
    transfer: BaseTransfer,
    parent: FlockNode,
    strategy: Strategy,
    module: FloxModule,
    module_state_dict: StateDict,
    dataset: Dataset | Subset | None = None,  # TODO: Cannot be `None`.
    **train_hyper_params,
) -> Result:
    pass
