![image](https://github.com/sail-sg/zero-bubble-pipeline-parallelism/assets/2740430/876bd529-c454-41ab-ad85-30dfb5e1c8fa)

# Quick Implementation on ZB-H1

This branch is a quick and straightforward implementation focusing on how to change 1F1B to ZB-H1 schedules.
The core code changes can be found in [this commit](https://github.com/sail-sg/zero-bubble-pipeline-parallelism/commit/95212f7000dca3d03dc518759020355cfdae231f) (`--sequence-parallel` not supported), involving less than 20 lines and a new file containing 60 lines.
To further support `--sequence-parallel`, please refer to [this commit](https://github.com/sail-sg/zero-bubble-pipeline-parallelism/commit/a84d634a5c0597b694f214ed8a72a8fdb0010bdb), involving about 80 lines in total.

In this quick implementation, we capture the weight gradient computation of linear layers (located in `megatron/core/tensor_parallel/layers.py`) and store them
in a `WeightGradStore` (see below). The execution of these computations is then deferred during scheduling, as handled in `megatron/core/pipeline_parallel/schedules.py`.

### `WeightGradStore`

```python
import queue
from megatron import get_args


class WeightGradStore:

    cache = []
    weight_grad_queue = queue.Queue()
    split_bw = True

    @classmethod
    def is_supported(cls):
        """If not supported, fallback to original schedule."""
        args = get_args()
        if args.pipeline_model_parallel_size <= 1:
            return False
        if args.virtual_pipeline_model_parallel_size is not None:
            return False
        if args.overlap_grad_reduce:
            # the logic of overlapping grad reduce should be changed
            return False
        if args.transformer_impl == 'transformer_engine':
            # hard to capture weight gradient computation for transformer_engine
            return False
        return True

    @classmethod
    def put(cls, total_input, grad_output, weight, func):
        if not cls.split_bw or not cls.is_supported():
            func(total_input, grad_output, weight.main_grad)
            return
        # Store the weight gradient computation of linear layers.
        cls.cache.append((total_input, grad_output, weight, func))

    @classmethod
    def flush(cls):
        if not cls.is_supported():
            return
        # Collect all stored computations during backward as a W.
        cls.weight_grad_queue.put(cls.cache)
        cls.cache = []

    @classmethod
    def pop(cls):
        if not cls.is_supported():
            return
        # Execute a single W.
        assert cls.weight_grad_queue.qsize() > 0
        stored_grads = cls.weight_grad_queue.get()
        for total_input, grad_output, weight, func in stored_grads:
            func(total_input, grad_output, weight.main_grad)

    @classmethod
    def pop_all(cls):
        # Execute all remaining W.
        remaining_qsize = cls.weight_grad_queue.qsize()
        for _ in range(remaining_qsize):
            cls.pop()
```

### Run the example
We provide a script to run a simple example. In this example, our implementation improves the throughput by about 9% against the 1F1B baseline without sacrificing anything.
```commandline
./examples/pretrain_zbh1.sh
```
