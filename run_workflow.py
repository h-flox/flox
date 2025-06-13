from flight.strategies.strategy import DefaultStrategy
from flight.system import flat_topology
from flight.workflow import FederationWorkflow


def main():
    topo = flat_topology(5)
    print(topo)

    workflow = FederationWorkflow(
        topo,
        DefaultStrategy(),
    )
    workflow.start()


if __name__ == "__main__":
    main()
