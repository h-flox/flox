## What are Strategies?
There are many different ways to perform Federated Learning. More specifically, there are many different ways to optimize the _global model_ being trained in an FL process with respect to many intricate challenges (e.g., system heterogeneity or statistical heterogeneity). In FLoX, a **Strategy** is an abstraction that implements one of these ways of optimizing the global model.

Some prominent examples from the literature of what we consider "Strategies" include (but are, of course, not limited to):

* Federated Stochastic Gradient Descent (`FedSGD`)[^fedavg]
* Federated Averaging (`FedAvg`)[^fedavg]
* FedAvg with Proximal Term (`FedProx`)[^fedprox]

## What _exactly_ do Strategies do?
In a nutshell, a lot. Federated Learning is a complex process with tasks being done on the worker nodes and the aggregator node(s). Thus, Strategies can touch a lot of different parts of the entire logic of an FL process.

### Model Parameter Aggregation
...

### Model Parameter Communication
...

#### Parameter Quantization/Compression
...

#### Parameter Encryption/Decryption
...

### Client Selection
...

### Local Training
...

## How Strategies are Run



[^fedavg]: McMahan, Brendan, et al. "Communication-efficient learning of deep networks from decentralized data." Artificial intelligence and statistics. PMLR, 2017. [(Link)](http://proceedings.mlr.press/v54/mcmahan17a/mcmahan17a.pdf)
[^fedprox]: Li, Tian, et al. "Federated optimization in heterogeneous networks." Proceedings of Machine learning and systems 2 (2020): 429-450. [(Link)](https://proceedings.mlsys.org/paper_files/paper/2020/file/1f5fe83998a09396ebe6477d9475ba0c-Paper.pdf)