# About FLoX (Federated Learning On funcX)

FLoX is a powerful, simple, and highly customizable *Federated Learning* (FL) framework. It is designed to simulate FL
workflows while also making it trivially easy to deploy them on real-world devices (e.g., Internet-of-Things and edge
devices).

## Federated Averaging (FedAvg)

A benchmark averaging algorithm.

$$\omega_{t+1} \triangleq \sum_{k=1}^{K} \frac{n_{k}}{n} \omega_{t+1}^{k}$$