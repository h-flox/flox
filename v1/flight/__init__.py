"""
**F**ederated **L**earning **I**n **G**eneral **H**ierarchical **T**opologies (FLIGHT)
is a lightweight framework for federated learning workflows for complex systems.
"""

from v1.flight.topologies import Topology
from v1.flight.topologies.utils import flat_topology

# from flight.fit import federated_fit

__all__ = ["Topology", "flat_topology"]
