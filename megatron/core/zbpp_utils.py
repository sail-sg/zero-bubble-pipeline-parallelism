import functools
import logging
import queue
from megatron.training import get_args, get_timers
from megatron.core import parallel_state
from megatron.core.distributed.finalize_model_grads import _allreduce_embedding_grads
from megatron.core.utils import get_model_config, get_attr_wrapped_model


class WeightGradStore:

    should_split_bw = False
    cache = []
    weight_grad_queue = [queue.Queue(), queue.Queue()]

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
        if not args.gradient_accumulation_fusion:
            return False
        if args.transformer_impl == 'transformer_engine':
            # hard to capture weight gradient computation for transformer_engine
            return False
        return True

    @classmethod
    def split_bw(cls):
        if not cls.is_supported():
            return False
        return cls.should_split_bw

    @classmethod
    def enable_split_bw(cls):
        cls.should_split_bw = True

    @classmethod
    def disable_split_bw(cls):
        cls.should_split_bw = False

    @classmethod
    def put(cls, weight, pre_func, func):
        assert cls.split_bw() == True
        # func(*pre_func(async_op=False))
        cls.cache.append((weight, pre_func, func))
        return

    @classmethod
    def flush(cls, chunk=0):
        cls.weight_grad_queue[chunk].put(cls.cache)
        cls.cache = []

    @classmethod
    def pop(cls, chunk=0):
        if cls.weight_grad_queue[chunk].qsize() > 0:
            stored_grads = cls.weight_grad_queue[chunk].get()
            for weight, pre_func, func in stored_grads:
                func(*pre_func(async_op=False))
        else:
            raise Exception("Pop empty queue.")

    @classmethod
    def clear(cls, model, chunk=0):
        weight_grad_tasks = []
        while cls.weight_grad_queue[chunk].qsize() > 0:
            stored_grads = cls.weight_grad_queue[chunk].get()
            if len(weight_grad_tasks) == 0:
                for _ in stored_grads:
                    weight_grad_tasks.append([])
            else:
                assert len(weight_grad_tasks) == len(stored_grads)
            for i, task in enumerate(stored_grads):
                weight_grad_tasks[i].append(task)
        # timers = get_timers()
        # weight_params = []
        # handles = []
        # if get_args().overlap_grad_reduce:
        #     handles += model.async_reduce_grad()

        # config = get_model_config(model)
        # # Do async all-reduce for embedding grads firstly, so that the rank 0 won't
        # # be blocked
        # embedding_handles = _allreduce_embedding_grads([model], config, async_op=True)
        # handles += embedding_handles

        for i in range(len(weight_grad_tasks)):
            tasks = weight_grad_tasks[i]
            param = None
            for j in range(len(tasks)):
                weight, pre_func, func = tasks[j]
                if param is None:
                    param = weight
                assert param is weight
                func(*pre_func(async_op=False))
                tasks[j] = None  # release memory
            # weight_params.append(param)
            # if get_args().overlap_grad_reduce:
            #     # All-reduce param grad here
            #     handles += model.async_reduce_grad(param)
            weight_grad_tasks[i] = None  # release memory

        # timers('wait_all_reduce', log_level=1).start(barrier=False)
        # for handle in handles:
        #     if handle is not None:
        #         handle.wait()
        # timers('wait_all_reduce').stop()
