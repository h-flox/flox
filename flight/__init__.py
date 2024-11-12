"""
**F**ederated **L**earning **I**n **G**eneral **H**ierarchical **T**opologies (FLIGHT)
is a lightweight framework for federated learning workflows for complex systems.
"""

from flight.federation.topologies import Topology
from flight.federation.topologies.utils import flat_topology
from flight.fit import federated_fit

__all__ = ["Topology", "flat_topology", "federated_fit"]
