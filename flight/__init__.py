"""

```mermaid
block-beta
    columns 3

    federation:3

    block:coordinator:2
        columns 2
        a b c d
    end

    block:strategy:1
        columns 1
        selection_policy
        aggregation_policy
        @Events...
    end

    aggr:1

    block:worker:2
        data[("Local\nData")]
    end
```

```mermaid
block-beta
    columns 3

    data_plane:3

    client
    aggregator

    block:worker
        data[("Local\nData")]
    end

    control_plane:3
```

"""
