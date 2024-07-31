"""
This module provides a Flight CLI for launching federations.

```sh title="Basic use of Flight CLI."
python3 -m flight.launch --config my-setup.yaml
```

```sh title="Configuring federation with Flight CLI args."
python3 -m flight.launch \
    --topology.kind hub-spoke \
    --topology.num_workers 10 \
    --dataset mnist \
    --output.results 'my_results.csv'
```
"""
