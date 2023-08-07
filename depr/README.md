## Callbacks

- ~~**[S]** on_model_init()~~
- **[S]** on_model_broadcast()
- **[W]** on_model_recv()
- **[W]** on_data_fetch()
- **[W]** on_model_fit()
- **[W]** on_model_send()
- **[S]** on_model_collate()
- **[S]** on_module_aggr()
- **[S]** on_model_eval()

***

```mermaid
classDiagram

AggrLogic--|>FedAvgAggr

class AggrLogic {
    on_model_init()*
    on_model_broadcast() 
    on_model_recv()
    on_module_aggr()
    on_model_eval()
}

class FederatedDataModule {
    prepare_distribution() None
}
```

## Synchronous vs. Asynchronous FL

```mermaid
sequenceDiagram
    participant Aggr
    actor Trainer1
    actor Trainer2
    
    Aggr->>Trainer1: share model
    activate Trainer1
    
    Aggr->>Trainer2: share model
    activate Trainer2
    
    Trainer1->>Aggr: locally-trained model
    deactivate Trainer1
    
    Aggr-->Trainer2: deadline
    Aggr->>Trainer2: kill
    deactivate Trainer2
    Trainer2->>Aggr: ack 
    
```