from collections.abc import Iterable, Mapping
from typing import TypeAlias

import torch

from flox.flock import FlockNode, FlockNodeID
from flox.flock.states import FloxAggregatorState, FloxWorkerState, NodeState
from flox.typing import StateDict

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

    registry = {}

    @classmethod
    def get_strategy(cls, name: str):
        """

        Args:
            name ():

        Returns:

        """
        name = name.lower()
        if name in cls.registry:
            return cls.registry[name]
        else:
            raise KeyError(f"Strategy name ({name=}) is not in the Strategy registry.")

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls.registry[cls.__name__.lower()] = cls

    ####################################################################################
    #                              AGGREGATOR CALLBACKS.                               #
    ####################################################################################

    def agg_before_round(self, state: FloxAggregatorState) -> None:
        """
        Some process to run at the start of a round.

        Args:
            state (FloxAggregatorState): The current state of the Aggregator FloxNode.
        """

    # @required
    def agg_param_aggregation(
        self,
        state: FloxAggregatorState,
        children_states: Mapping[FlockNodeID, NodeState],
        children_state_dicts: Mapping[FlockNodeID, StateDict],
        *args,
        **kwargs,
    ) -> StateDict:
        """

        Args:
            state (FloxAggregatorState):
            children_states (Mapping[FlockNodeID, NodeState]):
            children_state_dicts (Mapping[FlockNodeID, NodeState]):
            *args ():
            **kwargs ():

        Returns:
            StateDict
        """

    # @required
    def agg_worker_selection(
        self, state: FloxAggregatorState, children: Iterable[FlockNode], *args, **kwargs
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

    def agg_before_share_params(
        self, state: FloxAggregatorState, state_dict: StateDict, *args, **kwargs
    ) -> StateDict:
        """Callback before sharing parameters to child nodes.

        This is mostly done is modify the global model's StateDict. This can be done to encrypt the
        model parameters, apply noise, personalize, etc.

        Args:
            state (FloxAggregatorState): The current state of the aggregator.
            state_dict (StateDict): The global model's current StateDict (i.e., parameters) before
                sharing with workers.

        Returns:
            The global module StateDict.
        """
        return state_dict

    def agg_after_collect_params(
        self,
        state: FloxAggregatorState,
        children_states: Mapping[FlockNodeID, NodeState],
        children_state_dicts: Mapping[FlockNodeID, StateDict],
        *args,
        **kwargs,
    ) -> StateDict:
        """
        ...

        Args:
            state (FloxAggregatorState):
            children_states (Mapping[FlockNodeID, NodeState]): ...
            children_state_dicts (Mapping[FlockNodeID, StateDict]): ...
            *args ():
            **kwargs ():

        Returns:

        """

    ####################################################################################
    #                                WORKER CALLBACKS.                                 #
    ####################################################################################
    def wrk_on_before_train_step(self, state: FloxWorkerState, *args, **kwargs):
        """

        Args:
            state ():
            *args ():
            **kwargs ():

        Returns:

        """
        pass

    def wrk_on_after_train_step(
        self, state: FloxWorkerState, loss: Loss, *args, **kwargs
    ) -> Loss:
        """

        Args:
            state ():
            loss ():
            *args ():
            **kwargs ():

        Returns:

        """
        return loss

    def wrk_on_before_submit_params(
        self, state: FloxWorkerState, *args, **kwargs
    ) -> StateDict:
        """

        Args:
            state ():
            *args ():
            **kwargs ():

        Returns:

        """
        pass

    def wrk_on_recv_params(
        self, state: FloxWorkerState, params: StateDict, *args, **kwargs
    ):
        """

        Args:
            state ():
            params ():
            *args ():
            **kwargs ():

        Returns:

        """
        return params
