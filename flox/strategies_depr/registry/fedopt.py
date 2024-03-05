from flox.strategies_depr.base import Strategy


class FedOpt(Strategy):
    """
    Implementation of the FedOpt algorithm proposed by Reddi et al. (2020). It is implemented as a base class for the
    three specializations presented in the referenced paper, namely, ``FedAdaGrad``, ``FedAdam``, and ``FedYogi``.

    References:
        Reddi, S., Charles, Z., Zaheer, M., Garrett, Z., Rush, K., Konečný, J., ... & McMahan, H. B. (2020).
        Adaptive federated optimization. arXiv preprint arXiv:2003.00295.
    """

    def __init__(self):
        super().__init__()
        # TODO
