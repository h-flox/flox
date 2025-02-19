# TODO: Re-implement this entire FL strategy such that it no longer relies on
#  `TrainerStrategy` which is now deprecated.

# from __future__ import annotations
#
# import torch
#
# from flight.strategies.base import Strategy
#
# from ...federation.topologies.node import WorkerState
# from ...learning.types import LocalStepOutput
# from .fedavg import FedAvgWorker
# from .fedsgd import FedSGDAggr, FedSGDCoord
#
# DEVICE = "cpu"
#
#
# class FedProxTrainer(DefaultTrainerStrategy):
#     """The coordinator and its respective methods for 'FedProx'."""
#
#     def __init__(self, mu: float = 0.3):
#         self.mu = mu
#
#     def before_backprop(
#         self, state: WorkerState, loss: LocalStepOutput
#     ) -> LocalStepOutput:
#         """Callback to run before backpropagation.
#
#         Args:
#             state (WorkerState): The state of the current node.
#             loss (LocalStepOutput): The calculated loss associated with the
#                 current node.
#
#         Returns:
#             The updated local step output (or loss) associated with the current node.
#         """
#         global_model = state.pre_module
#         local_model = state.module
#         assert global_model is not None
#         assert local_model is not None
#
#         # TODO: Re-implement this, require that this Strategy only work with PyTorch,
#         #       or implement an abstraction for sending to different devices.
#         assert isinstance(global_model, torch.nn.Module)
#         assert isinstance(local_model, torch.nn.Module)
#
#         global_model = global_model.to(DEVICE)
#         local_model = local_model.to(DEVICE)
#
#         params = list(local_model.get_params().values())
#         params0 = list(global_model.get_params().values())
#
#         proximal_diff = torch.tensor(
#             [
#                 torch.sum(torch.pow(params[i] - params0[i], 2))
#                 for i in range(len(params))
#             ]
#         )
#         proximal_term = torch.sum(proximal_diff)
#         proximal_term = proximal_term * self.mu / 2
#
#         proximal_term = proximal_term.to(DEVICE)
#
#         loss += proximal_term
#         return loss
#
#
# class FedProx(Strategy):
#     """
#     Implementation of FedAvg with Proximal Term.
#
#     This strategy extends ``FedAvg`` and differs from it by computing a
#     "proximal term"
#     and adding it to the computed loss during the training step before doing
#     backpropagation. This proximal term is the norm difference between the parameters
#     of the global model and the worker's locally-updated model. This proximal term is
#     used to make aggregation less sensitive to harshly heterogeneous (i.e., non-iid)
#     data distributions across workers.
#
#     More information on the proximal term and its definition can be found in the
#     docstring for ``FedProx.wrk_after_train_step()`` and in the reference below.
#
#     References:
#         Li, Tian, et al. "Federated optimization in heterogeneous networks."
#         *Proceedings of Machine learning and systems* 2 (2020): 429-450.
#     """
#
#     def __init__(
#         self,
#         mu: float = 0.3,
#         participation: float = 1.0,
#         probabilistic: bool = False,
#         always_include_child_aggregators: bool = False,
#     ):
#         super().__init__(
#             coord_strategy=FedSGDCoord(
#                 participation,
#                 probabilistic,
#                 always_include_child_aggregators,
#             ),
#             aggr_strategy=FedSGDAggr(),
#             worker_strategy=FedAvgWorker(),
#             trainer_strategy=FedProxTrainer(mu=mu),
#         )
