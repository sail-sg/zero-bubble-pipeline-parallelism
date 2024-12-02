import functools
import logging
import os
import queue
from contextlib import contextmanager

from megatron.training import get_args, get_timers
from megatron.core import parallel_state
from megatron.core.distributed.finalize_model_grads import _allreduce_embedding_grads
from megatron.core.utils import get_model_config, get_attr_wrapped_model


def add_zero_bubble_args(parser):
    group = parser.add_argument_group(title='zero bubble')
    group.add_argument('--enable-zb-runtime', action='store_true',
                       help='Use an unified runtime supporting zero-bubble and other schedules.',
                       dest='enable_zb_runtime')
    group.add_argument('--no-pre-communication-optimization', action='store_false',
                       help='By default zb runtime dispatches a tiny communication before the real communication to optimize computation',
                       dest='pre_communication_optimization')
    group.add_argument('--zero-bubble-pipeline-timers-start-iter',
                       type=int, default=100,
                       help='The starting iteration that start timers for auto scheduling of zero-bubble pipeline parallel')
    group.add_argument('--zero-bubble-pipeline-timers-end-iter',
                       type=int, default=110,
                       help='The starting iteration that stop timers for auto scheduling of zero-bubble pipeline parallel')
    group.add_argument('--zero-bubble-max-pending-backward',
                       type=str, default="auto",
                       help='Maximum number of pending backward for zero-bubble. E.g. when number of stages are 8, setting to 16 will use zb2p and setting to 8 will use zb1p. Setting to auto will enable adaptive memory limit')
    group.add_argument('--zero-bubble-adaptive-memory-limit-percentile',
                       type=int, default=85,
                       help='Adaptively set the memory limit of ZB schedules so all pytorch mem allocations will use up to this percentile of total GPU memory. Currently ZBV is not supported.')
    group.add_argument('--enable-optimizer-post-validation',
                       action='store_true',
                       help='enable post validation for optimizer step',
                       dest='enable_optimizer_post_validation')
    group.add_argument('--enable-exactly-numeric-match',
                       action='store_true',
                       help='whether to make optimizer post validation exactly numeric match baseline',
                       dest='enable_exactly_numeric_match')
    group.add_argument('--enable-zero-bubble', action='store_true',
                       help='Use zero bubble pipeline.',
                       dest='enable_zero_bubble')
    group.add_argument('--zero-bubble-v-schedule', action='store_true',
                       help='Use zero bubble v schedule pipeline. This method achieves zero-bubble without more memory overhead',
                       dest='zero_bubble_v_schedule')
    group.add_argument('--zero-bubble-v-schedule-mem-setup', type=str,
                       default='zb',
                       help='Use zero bubble v schedule pipeline with memory setup.')
    group.add_argument('--enable-1f1b-v', action='store_true',
                       help='Use 1F1B V schedule.',
                       dest='enable_1f1b_v')
    group.add_argument('--allow-padding-num-layers', action='store_true',
                       help='Allow padding num_layers for pipeline parallelism',
                       dest='allow_padding_num_layers')
    group.add_argument('--profile-memory-iter',
                       type=int, default=-1,
                       help='The iteration to profile memory.')
    group.add_argument('--interleave-group-size',
                       type=int, default=0,
                       help='Set interleave group size')
    group.add_argument('--offload-chunk-num',
                       type=int, default=0,
                       help='offload chunk number')
    group.add_argument('--offload-time',
                       type=float, default=1.0,
                       help='offload time cost.')
    group.add_argument('--offload-overlap-sr', action='store_true',
                       help='overlap save and resume in offload')
    return parser


def validate_arguments(args):
    assert args.untie_embeddings_and_output_weights == True, "Not supported for code cleanness"
    assert args.defer_embedding_wgrad_compute == False, "The original code seems incorrect"

    if args.zero_bubble_v_schedule \
            or args.enable_zero_bubble \
            or args.enable_1f1b_v \
            or args.num_seq_splits > 1 \
            or args.interleave_group_size > 0 \
            or args.enable_zb_runtime:
        args.enable_zb_runtime = True
        if not args.gradient_accumulation_fusion:
            raise RuntimeError("gradient-accumulation-fusion should be True for zb runtime.")

    if args.pre_communication_optimization:
        if not args.enable_zb_runtime or not args.overlap_p2p_comm:
            raise RuntimeError(
                "pre-communication only works with --enable-zb-runtime and without --no-overlap-p2p-communication")
    if args.enable_zb_runtime and args.overlap_p2p_comm:
        # Our tests are done on setting CUDA_DEVICE_MAX_CONNECTIONS = 8
        # A smaller number larger than 4 may also work.
        # For simplicity just use 8 as minimum here.
        cuda_device_max_conn = int(os.environ.get('CUDA_DEVICE_MAX_CONNECTIONS') or 8)
        if cuda_device_max_conn < 8:
            raise RuntimeError(
                "Set CUDA_DEVICE_MAX_CONNECTIONS >= 8 for overlap-p2p-communication")

    # if not args.overlap_p2p_comm:
    #     if os.environ.get('CUDA_DEVICE_MAX_CONNECTIONS') != "1":
    #         raise RuntimeError(
    #             "CUDA_DEVICE_MAX_CONNECTIONS must be 1 for batching communication")

    # TODO: validate more
    if args.zero_bubble_v_schedule or args.enable_1f1b_v:
        assert args.num_layers % args.transformer_pipeline_model_parallel_size == 0, \
            'number of layers should be divisible by the pipeline parallel size'
        num_layers_per_pipeline_stage = args.num_layers // args.transformer_pipeline_model_parallel_size
        assert num_layers_per_pipeline_stage % 2 == 0, \
            'zero bubble v and 1f1b v schedule requires number of layers per pipeline stage to be even'
        assert args.num_layers_per_virtual_pipeline_stage is None, \
            'num_layers_per_virtual_pipeline_stage should not be set with zero bubble v and 1f1b v schedule'
        args.virtual_pipeline_model_parallel_size = 2
        args.num_layers_per_virtual_pipeline_stage = num_layers_per_pipeline_stage // 2
        assert args.virtual_pipeline_model_parallel_size == 2

    if args.zero_bubble_v_schedule:
        args.enable_zero_bubble = True
        assert args.zero_bubble_v_schedule_mem_setup in {'min', 'half', 'zb'}

    if args.enable_1f1b_v:
        assert args.pipeline_model_parallel_size > 1, "1f1b-v must be enabled with pipeline parallelism"
        assert not args.enable_zero_bubble, "cannot enable zero bubble for 1f1b-v"
        assert not args.enable_optimizer_post_validation, "cannot enable post validation for 1f1b-v"

    if args.enable_zero_bubble:
        if args.use_distributed_optimizer:
            assert not args.overlap_param_gather, "the original code somehow doesn't work"
            assert not args.overlap_grad_reduce, "not supported yet because we didn't verify the correctness"
        assert args.pipeline_model_parallel_size > 1, "zero bubble must be enabled with pipeline parallelism"
        if args.enable_optimizer_post_validation:
            assert args.fp16, "zero bubble post validation"
        if args.zero_bubble_max_pending_backward == 'auto':
            assert args.zero_bubble_adaptive_memory_limit_percentile > 0
        else:
            args.zero_bubble_max_pending_backward = int(args.zero_bubble_max_pending_backward)
    else:
        args.enable_optimizer_post_validation = False


class WeightGradStore:

    should_split_bw = False
    cache = []
    weight_grad_queue = None  # lazy init

    @classmethod
    def lazy_init(cls):
        if cls.weight_grad_queue is not None:
            return
        # Lazy init to make sure parallel_state and get_args() have been initialized.
        num_chunks = parallel_state.get_virtual_pipeline_model_parallel_world_size() or 1
        # chunk id => seq id => Queue
        cls.weight_grad_queue = [[queue.Queue() for _ in range(get_args().num_seq_splits)] for _ in range(num_chunks)]

    @classmethod
    def is_supported(cls):
        """If not supported, fallback to original schedule."""
        args = get_args()
        if args.pipeline_model_parallel_size <= 1:
            return False
        # if args.virtual_pipeline_model_parallel_size is not None:
        #     return False
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
    @contextmanager
    def set_split_bw(cls, enabled: bool):
        prev = cls.should_split_bw
        cls.should_split_bw = enabled
        try:
            yield
        finally:
            cls.should_split_bw = prev

    @classmethod
    def put(cls, weight, pre_func, func):
        assert cls.split_bw()
        # func(*pre_func(async_op=False))
        cls.cache.append((weight, pre_func, func))
        return

    @classmethod
    def queue_size(cls, chunk=0, seq_split_idx=0):
        cls.lazy_init()
        return WeightGradStore.weight_grad_queue[chunk][seq_split_idx].qsize()

    @classmethod
    def flush(cls, chunk=0, seq_split_idx=0):
        cls.lazy_init()
        # Or W later will consume empty computation and leak the non-empty computation.
        if not cls.split_bw():
            assert len(cls.cache) == 0
            return
        cls.weight_grad_queue[chunk][seq_split_idx].put(cls.cache)
        cls.cache = []

    @classmethod
    def pop(cls, chunk=0, seq_split_idx=0):
        cls.lazy_init()
        if cls.weight_grad_queue[chunk][seq_split_idx].qsize() > 0:
            stored_grads = cls.weight_grad_queue[chunk][seq_split_idx].get()
            for weight, pre_func, func in stored_grads:
                func(*pre_func(async_op=False))
        else:
            rank = parallel_state.get_pipeline_model_parallel_rank()
            raise Exception(f"Pop empty queue. rank {rank}")

    @classmethod
    def assert_empty(cls):
        rank = parallel_state.get_pipeline_model_parallel_rank()
        assert len(cls.cache) == 0, f"cache is not empty. rank {rank}"
        if cls.weight_grad_queue is None:
            return
        for chunk, chunk_q in enumerate(cls.weight_grad_queue):
            for seq, seq_q in enumerate(chunk_q):
                assert seq_q.empty(), f"Queue is not empty chunk {chunk} seq {seq} rank {rank}. len {seq_q.qsize()}"

    @classmethod
    def clear(cls, model, chunk=0, seq_split_idx=0):
        cls.lazy_init()
        weight_grad_tasks = []
        while cls.weight_grad_queue[chunk][seq_split_idx].qsize() > 0:
            stored_grads = cls.weight_grad_queue[chunk][seq_split_idx].get()
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
                assert param.storage().data_ptr() == weight.storage().data_ptr()
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

class RecomputeStore:

    cache = []
    recompute_queue = None
    recompute_flag = False

    @classmethod
    def lazy_init(cls):
        if cls.recompute_queue is not None:
            return
        # Lazy init to make sure parallel_state and get_args() have been initialized.
        num_chunks = parallel_state.get_virtual_pipeline_model_parallel_world_size() or 1
        # chunk id => Queue
        cls.recompute_queue = [queue.Queue() for _ in range(num_chunks)]

    @classmethod
    @contextmanager
    def set_recompute_flag(cls, enabled: bool):
        prev = cls.recompute_flag
        cls.recompute_flag = enabled
        try:
            yield
        finally:
            cls.recompute_flag = prev

    @classmethod
    def should_recompute(cls):
        return cls.recompute_flag

    @classmethod
    def put(cls, func):
        assert cls.recompute_flag
        cls.lazy_init()
        cls.cache.append(func)

    @classmethod
    def flush(cls):
        if not cls.recompute_flag:
            assert len(cls.cache) == 0
            return
        cls.lazy_init()
        chunk = parallel_state.get_virtual_pipeline_model_parallel_rank()
        cls.recompute_queue[chunk].put(cls.cache)
        cls.cache = []

    @classmethod
    def pop(cls):
        cls.lazy_init()
        chunk = parallel_state.get_virtual_pipeline_model_parallel_rank()
        recompute_funcs = cls.recompute_queue[chunk].get()
        for func in recompute_funcs:
            func()
