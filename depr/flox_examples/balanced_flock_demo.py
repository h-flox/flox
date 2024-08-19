import matplotlib.pyplot as plt

from flox.federation.topologies import balanced_hierarchical_topology

if __name__ == "__main__":
    tree = balanced_hierarchical_topology(10, 3)
    tree.draw(with_labels=False)
    plt.show()
