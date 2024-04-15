import matplotlib.pyplot as plt

from flox.flock.factory import created_balanced_hierarchical_flock

if __name__ == "__main__":
    tree = created_balanced_hierarchical_flock(10, 3)
    tree.draw(with_labels=False)
    plt.show()
