Welcome to FLoX's documentation!
================================

.. module:: flox

  Hello world!!

.. toctree::
    :caption: Contents:
    :maxdepth: 0

   getting_started
   usage/quickstart
   api/flox


Getting Started
---------------
FLoX is a simple, highly customizable, and easy-to-deploy framework for launching Federated Learning processes across a
decentralized network. It is designed to simulate FL workflows while also making it trivially easy to deploy them on
real-world devices (e.g., Internet-of-Things and edge devices). Built on top of **Globus Compute** (formerly known as
*funcX*), FLoX is designed to run on anything that can be started as a Globus Compute Endpoint.


Installation
^^^^^^^^^^^^
The package can be found on PyPI:

.. code-block:: bash

    pip install pyflox

What can FLoX do?
-----------------

FLoX is supports several state-of-the-art approaches for FL processes, including hierarchical and asynchronous FL.


Usage
-----

FLoX is a simple, highly-customizable, and easy-to-deploy framework for hierarchical, multi-tier federated learning systems built on top of the Globus Compute platform.

.. code-block:: python
  :linenos:

  from flox import Flock, federated_fit
  from torch import nn

  class MyModule(nn.Module):
      ...

  flock = Flock.from_yaml("my_flock.yaml")
  federated_fit(
    module=MyModule(),
    flock=flock,
    strategy="fedavg",
    strategy_params={"participation_frac": 0.5},
    where="local",
    logger="csv",
    log_out="my_results.csv"
  )


Indices and Tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`