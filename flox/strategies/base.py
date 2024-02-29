from __future__ import annotations

import abc
import typing

if typing.TYPE_CHECKING:
    import torch

    from typing import Iterable, MutableMapping, TypeAlias
    from flox.flock import FlockNode, FlockNodeID
    from flox.flock.states import WorkerState, AggrState, NodeState
    from flox.nn.typing import StateDict

    Loss: TypeAlias = torch.Tensor


class Strategy:
    """Base class for the logical blocks of a FL process.

    A ``Strategy`` in FLoX is used to implement the logic of an FL process. A ``Strategy`` provides
    a number of callbacks which can be overridden to inject pieces of logic throughout the FL process.
    Some of these callbacks are run on the aggregator nodes while others are run on the worker nodes.

    It is _**highly**_ encouraged that you read [What Do Strategies Do](/getting_started/strategies/what/)
    to better understand how the callbacks included in a Strategy interact with one another and when
    they are run in an FL process.
    """

    __metaclass__ = abc.ABCMeta

    registry: MutableMapping[str, type["Strategy"]] = {}
    """..."""

    def __new__(cls, *args, **kwargs):
        if cls.__class__ == (Strategy,):
            raise TypeError(f"Abstract class {cls.__name__} cannot be instantiated.")
        return super(Strategy, cls).__new__(cls, *args, **kwargs)

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls.registry[cls.__name__.lower()] = cls

    @classmethod
    def get_strategy(cls, name: str) -> type["Strategy"]:
        """
        Pulls a strategy class implementation from the registry by its name.

        Notes:
            All names are lower-cased (e.g., the name for `FedAvg` is "fedavg"). Thus, any
            provided argument for `name` is lower-cased via `name = name.lower()`.

        Args:
            name (str): The name of the strategy implementation to pull from the registry.

        Returns:
            Strategy class.
        """
        name = name.lower()
        if name in cls.registry:
            return cls.registry[name]
        else:
            raise KeyError(f"Strategy name ({name}) is not in the Strategy registry.")

    ####################################################################################
    #                                CLIENT CALLBACKS.                                 #
    ####################################################################################

    def cli_get_node_statuses(self):
        """
        Followup callback upon getting status updates from all of the nodes in the Flock.
        """

    def cli_worker_selection(
        self, state: AggrState, children: Iterable[FlockNode], **kwargs
    ) -> Iterable[FlockNode]:
        """

        Args:
            state ():
            children ():
            *args ():
            **kwargs ():

        Returns:
            List of selected nodes that are children of the aggregator.
        """
        return children

    def cli_before_share_params(
        self, state: AggrState, state_dict: StateDict, **kwargs
    ) -> StateDict:
        """Callback before sharing parameters to child nodes.

        This is mostly done is modify the global model's StateDict. This can be done to encrypt the
        model parameters, apply noise, personalize, etc.

        Args:
            state (AggrState): The current state of the aggregator.
            state_dict (StateDict): The global model's current StateDict (i.e., parameters) before
                sharing with workers.

        Returns:
            The global global_module StateDict.
        """
        return state_dict

    ####################################################################################
    #                              AGGREGATOR CALLBACKS.                               #
    ####################################################################################

    def agg_before_round(self, state: AggrState) -> None:
        """
        Some process to run at the start of a round.

        Args:
            state (AggrState): The current state of the Aggregator FloxNode.
        """
        raise NotImplementedError

    def agg_param_aggregation(
        self,
        state: AggrState,
        children_states: MutableMapping[FlockNodeID, NodeState],
        children_state_dicts: MutableMapping[FlockNodeID, StateDict],
        *args,
        **kwargs,
    ) -> StateDict:
        """

        Args:
            state (AggrState):
            children_states (Mapping[FlockNodeID, NodeState]):
            children_state_dicts (Mapping[FlockNodeID, NodeState]):
            *args ():
            **kwargs ():

        Returns:
            StateDict
        """
        raise NotImplementedError

    ####################################################################################
    #                                WORKER CALLBACKS.                                 #
    ####################################################################################

    def wrk_on_recv_params(self, state: WorkerState, params: StateDict, **kwargs):
        """

        Args:
            state ():
            params ():
            *args ():
            **kwargs ():

        Returns:

        """
        return params

    def wrk_before_train_step(self, state: WorkerState, **kwargs):
        """

        Args:
            state ():
            *args ():
            **kwargs ():

        Returns:

        """
        raise NotImplementedError()

    def wrk_after_train_step(self, state: WorkerState, loss: Loss, **kwargs) -> Loss:
        """

        Args:
            state ():
            loss ():
            *args ():
            **kwargs ():

        Returns:

        """
        return loss

    def wrk_before_submit_params(self, state: WorkerState, **kwargs) -> StateDict:
        """

        Args:
            state ():
            *args ():
            **kwargs ():

        Returns:

        """
        raise NotImplementedError()
