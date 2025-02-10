# Generic Pipeline Parallel Runtime

This codebase introduced a generic runtime system for Pipeline Parallelism that simplifies implementing new schedules.

**NOTE**: the current codebase does not support `transformer-engine`.
Need to specify the following options when running Megatron-LM:
```
--use-legacy-models
--transformer-impl local
--enable-zb-runtime
```

### Overall Design
The original implementation of Megatron-LM Pipeline Parallel in `megatron/core/pipeline_parallel/schedules.py` mixes both the scheduling and execution, making it difficult to maintain as new schedules were introduced.
To address this, we decouple the execution part to a runtime module that accepts schedules to run.

The new implementation is structured into 3 components:
- **schedules**: Specifies schedules for computation like F, B, W.
- **post-processing**: Automatically add communication schedules (and offloading if needed).
- **runtime**: Executes schedules after post-processing.

### Adding a New Schedule
The [1F1B](../../megatron/core/pipeline_parallel/zerobubble/scheduler/basic1f1b.py)
serves as a good example to add new schedules:
- Define a schedule function
- Add the schedule function to runtime

#### (1) Define a schedule function

```python
from megatron.core.pipeline_parallel.zerobubble.scheduler.graph import GraphConfig, ScheduledNode, F, BW

def create_schedule(config: GraphConfig):
    # Return a 2d array of ScheduledNode
    local_order = [[] for _ in config.n_stages]
    for stage in range(config.n_stages):
        ...
        local_order[stage].append(
            ScheduledNode(
                type=F,
                stage=stage,
                microbatch=microbatch,
                layer_group_idx=stage,
            )
        )
        ...
    return local_order
```

`ScheduledNode` specifies all the operation node in runtime,
including computation, communication, offloading.
In the schedule function, we only need to specify computation nodes.

Some key parameters in `ScheduledNode`:
- `type`: computaion type like F, B, W, BW
  - F: forward
  - B: backward only for activation
  - W: backward only for parameter
  - BW: backward for both activation and parameter
- `layer_group_idx`: Pipeline Parallelism divides the transformer layers into groups.
`layer_group_idx` specifies the index of the group.
When Interleaved and V schedules are not enabled, This is simply the same as `stage`.
This helps other modules like **post-processing** to determine the dependencies between ScheduledNode. 

#### (2) Add schedule function to runime
In the `get_zero_bubble_forward_backward_func` function in [megatron/core/pipeline_parallel/zerobubble/runtime.py](../../megatron/core/pipeline_parallel/zerobubble/runtime.py),
add the following call to schedule function:
```python
def get_zero_bubble_forward_backward_func():
    ...
    if get_args().enable_my_new_schedule:
        def scheduler(nstages, nmb, f, b, w, c, f_mem, b_mem, w_mem, mem_limit):
            f_mid = avg_then_mid(f)
            b_mid = avg_then_mid(b)
            w_mid = avg_then_mid(w)
            config = zb.GraphConfig.basic_config(
                f=f_mid,
                b=b_mid,
                w=w_mid,
                n_stages=nstages,
                n_micro=nmb,
                # In interleaved schedule,
                # this should be the number of chunks.
                max_chunks=1,
            )
            local_order = myschedule.create_schedule(config)
            # run post-processing
            ret = run_schedule_passes(config, local_order)
            return ret

        global_zb_runtime = get_zb_runtime_instance()
        forward_backward_func = wrapped_auto_schedule_forward_backward_func(global_zb_runtime, scheduler=scheduler)
        return forward_backward_func
    ...
```

# Runtime

#### CUDA_DEVICE_MAX_CONNECTIONS > 1
When either **overlap-p2p-communication** or Tensor Parallel is enabled,
the original Megatron-LM implementation requires `CUDA_DEVICE_MAX_CONNECTIONS=1`.
This setting ensures that communication kernels are scheduled before computation kernels,
allowing communication and computation to overlap effectively.
Without this, if the computation kernel is scheduled first, it can monopolize all Streaming Multiprocessor (SM) resources,
leaving no room for communication kernels to execute concurrently.

Alternatively, this behavior can also be achieved by prioritizing the communication stream while maintaining a larger CUDA_DEVICE_MAX_CONNECTIONS value.
This approach will be utilized in the Pre-communication Optimization strategy.

#### Pre-communication Optimization
While overlapping computation and communication helps mask communication latency,
prolonged communication waits can still degrade computational performance.
To mitigate this, we introduce a lightweight send-recv operation prior to the primary send-recv phase.
This preliminary operation minimizes the computational impact by reserving only minimal streaming multiprocessor (SM) resources for synchronization,
thereby reducing contention during subsequent communication.

For this optimization to work effectively,
each send-recv operation must execute independently and in parallel.
This requires enabling CUDA_DEVICE_MAX_CONNECTIONS (set to a value greater than 1),
which allows the CUDA runtime to concurrently schedule multiple communication kernels.
