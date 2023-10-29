# Globus Compute

## Globus Compute: Federated Function-as-a-Service

### Quick, High-Level Introduction
[**Globus Compute** ](https://funcx.readthedocs.io/en/latest/index.html)(formerly known as funcX) is a distributed _Function as a Service_ (FaaS) platform that enables flexible, scalable, and high performance remote function execution. Unlike centralized FaaS platforms, Globus Compute allows users to execute functions on heterogeneous remote computers, from laptops to campus clusters, clouds, and supercomputers.

**Globus Compute** is composed of 2 core components:

1. The Globus Compute cloud-hosted service provides an available, reliable, and secure interface for registering, sharing, and executing functions on remote endpoints. It implements a fire-and-forget model via which the cloud service is responsible for securely communicating with endpoints to ensure functions are successfully executed.
2. Globus Compute endpoints transform existing laptops, clouds, clusters, and supercomputers into function serving systems. Endpoints are registered by installing the Globus Compute endpoint software and configuring it for the target system.


### Why Globus Compute?
Managing scattered, decentralized infrastructure (e.g., edge and IoT devices) is nontrivial. As a result, setting up real-world FL systems is often in accessible to practitioners and researchers alike. Globus Compute solves this problem by employing a fire-and-forget framework to simplify the execution of functions on remote devices. Globus Compute serves as a simple, yet powerful, platform for launching decentralized computation with minimal management of the network layer.

In short, if your devices have Internet connectivity, can run `pip install globus-compute-sdk`, and have sufficient compute to train neural networks, then they are ready for FLoX.

