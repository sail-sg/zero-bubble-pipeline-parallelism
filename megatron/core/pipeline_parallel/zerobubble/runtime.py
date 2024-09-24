import contextlib
import copy
import itertools
from dataclasses import dataclass
from typing import Iterator, Tuple, List, Union, Callable, Any, Optional

import torch

from megatron.core.pipeline_parallel.p2p_communication import _communicate
from megatron.core.pipeline_parallel.zerobubble.scheduler.graph import F, B, BW, W, FuncType
from megatron.training import get_args, print_rank_0
from megatron.core.num_microbatches_calculator import get_num_microbatches
from megatron.core.pipeline_parallel.zerobubble.scheduler import run_schedule_passes, seq1f1b, vpp
from megatron.core.utils import get_model_config, get_model_type, get_model_xattn
from megatron.core import parallel_state
from megatron.core.parallel_state import (
    get_pipeline_model_parallel_group,
    get_pipeline_model_parallel_next_rank,
    get_pipeline_model_parallel_prev_rank,
)
from megatron.core.pipeline_parallel.schedules import (
    recv_forward,
    send_forward,
    recv_backward,
    send_backward,
    deallocate_output_tensor,
    forward_step,
    backward_step,
    get_tensor_shapes,
)
from megatron.core.pipeline_parallel.zerobubble.scheduler import ScheduledNode, zbv_greedy, zbv, zb, CommDirection, \
    v1f1b
from megatron.core.pipeline_parallel.zerobubble.timer import ScheduleTimers
from megatron.core.zbpp_utils import WeightGradStore
from megatron.core.timers import Timer
from megatron.training.training import RollbackDataIteratorWrapper
from megatron.training.utils import is_second_last_pipeline_stage


AUTO_SCHEDULE_COMMUNICATION_TYPES = {
    FuncType.RECV_FORWARD,
    FuncType.RECV_BACKWARD,
    FuncType.SEND_FORWARD,
    FuncType.SEND_BACKWARD,
}


@dataclass
class TrainingIterationConfig:
    run_timer: bool
    schedules: List[ScheduledNode]
    forward_step_func: Callable[..., Any]
    data_iterator: List[RollbackDataIteratorWrapper]
    model: List[torch.nn.Module]
    model_type: Any
    # config of model
    config: Any
    num_microbatches: int
    forward_only: bool
    collect_non_loss_data: bool

    no_sync_func: Callable
    tensor_shape: Tuple
    recv_tensor_shapes: List
    send_tensor_shapes: List


class SpQueue:
    """A queue of a stack"""

    def __init__(self, num_seq_splits):
        # Using two queues for safety of abusing.
        self.ready_queue = []
        self.tmp_stack = []
        self.num_seq_splits = num_seq_splits

    def push(self, tensor):
        self.tmp_stack.append(tensor)
        if len(self.tmp_stack) == self.num_seq_splits:
            self.ready_queue.append(self.tmp_stack)
            self.tmp_stack = []

    def pop(self):
        assert self.ready_queue
        ret = self.ready_queue[0].pop(-1)
        if not self.ready_queue[0]:
            self.ready_queue.pop(0)
        return ret


class TrainingIteration:
    class Buffers:
        def __init__(self):
            # two dim array, first dim is the model chunk, second dim is the microbatch queue
            num_chunks = get_virtual_pipeline_number()
            self.input_tensors = [SpQueue(get_args().num_seq_splits) for _ in range(num_chunks)]
            self.output_tensors = [SpQueue(get_args().num_seq_splits) for _ in range(num_chunks)]
            self.total_num_tokens = torch.tensor(0, dtype=torch.int).cuda()
            chunk_buf = [[] for _ in range(get_args().num_seq_splits)]
            buf = [copy.deepcopy(chunk_buf) for _ in range(num_chunks)]
            self.send_forward_buffer = copy.deepcopy(buf)
            self.recv_forward_buffer = copy.deepcopy(buf)
            self.send_backward_buffer = copy.deepcopy(buf)
            self.recv_backward_buffer = copy.deepcopy(buf)
            self.forward_data_store = []
            self.local_send_forward_buffer = [[] for _ in range(get_args().num_seq_splits)]
            self.local_send_backward_buffer = [[] for _ in range(get_args().num_seq_splits)]

        def buffer_map(self, node: ScheduledNode):
            return {
                FuncType.SEND_FORWARD: self.send_forward_buffer[node.chunk][node.seq_split_idx],
                FuncType.RECV_FORWARD: self.recv_forward_buffer[node.chunk][node.seq_split_idx],
                FuncType.SEND_BACKWARD: self.send_backward_buffer[node.chunk][node.seq_split_idx],
                FuncType.RECV_BACKWARD: self.recv_backward_buffer[node.chunk][node.seq_split_idx],
            }[node.type]

    class States:
        def __init__(self):
            num_chunks = get_virtual_pipeline_number()
            self.send_handles = []
            self.w_clear_run = [False] * num_chunks
            # map of {direction -> {node, shape}}
            self.communication_batch = {
                'SEND_NEXT': [],
                'RECV_NEXT': [],
                'SEND_PREV': [],
                'RECV_PREV': [],
            }
            self.it = 0

    def __init__(
        self, iteration_config: TrainingIterationConfig,
        iteration_id: int,
    ):
        self.no_sync_context = None
        self.iteration_config = iteration_config
        self.states = TrainingIteration.States()
        self.buffers = TrainingIteration.Buffers()
        self.iteration_id = iteration_id

    def update_config(
        self, iteration_config,
    ):
        """
        The config will be updated on each run of the iteration.
        """
        self.iteration_config = iteration_config

    def reset(self):
        self.states = TrainingIteration.States()
        self.buffers = TrainingIteration.Buffers()

    def _free_buffers(self):
        self.buffers = TrainingIteration.Buffers()

    def run(self):
        it = self.states.it
        conf = self.iteration_config
        multi_chunks = get_virtual_pipeline_number() > 1
        bufs = self.buffers

        WeightGradStore.assert_empty()
        self.disable_grad_sync()

        if get_args().profile:
            torch.cuda.nvtx.range_push(f'iter_{torch.distributed.get_rank()}_{ScheduleTimers.iter_counter}')

        while it < len(conf.schedules):
            scheduled_node = conf.schedules[it]
            if multi_chunks:
                parallel_state.set_virtual_pipeline_model_parallel_rank(scheduled_node.chunk)
            if scheduled_node.type.is_post_validation_related():
                # Ignore post validation nodes.
                pass
            elif scheduled_node.type in AUTO_SCHEDULE_COMMUNICATION_TYPES:
                next_is_comm = it + 1 < len(conf.schedules) and conf.schedules[
                    it + 1].type in AUTO_SCHEDULE_COMMUNICATION_TYPES
                next_compute = list(filter(lambda x: x.type.is_computation(), conf.schedules[it + 1:]))
                next_compute = next_compute[0] if len(next_compute) > 0 else None
                self.add_communication(scheduled_node, next_is_comm, next_compute)
            elif scheduled_node.type == F:
                self.schedule_f(scheduled_node)
            elif scheduled_node.type == B:
                self.schedule_b(scheduled_node)
            elif scheduled_node.type == BW:
                self.schedule_bw(scheduled_node)
            elif scheduled_node.type == W:
                non_w_pending = any([node.type != W for node in conf.schedules[it + 1:]])
                self.schedule_w(scheduled_node, non_w_pending)
            else:
                raise ValueError(f"Unknown node type {scheduled_node.type}")
            it += 1
        self.states.it = it

        self.wait_for_comm()

        if get_args().profile:
            torch.cuda.nvtx.range_pop()  # iter

        if not conf.forward_only:
            # Launch any remaining grad reductions
            if self.no_sync_context is not None:
                self.enable_grad_sync()

            if conf.config.finalize_model_grads_func is not None:
                assert isinstance(conf.model, list), "model should be a list"
                # Finalize model grads (perform full grad all-reduce / reduce-scatter for
                # data parallelism, layernorm all-reduce for sequence parallelism, and
                # embedding all-reduce for pipeline parallelism).
                conf.config.finalize_model_grads_func(
                    conf.model, bufs.total_num_tokens if conf.config.calculate_per_token_loss else None
                )

            if get_args().zero_bubble_pipeline_timers_end_iter == ScheduleTimers.iter_counter:
                ScheduleTimers.concluded = True

        WeightGradStore.assert_empty()
        return bufs.forward_data_store

    def run_until_post_validation(self, optimizer):
        conf = self.iteration_config
        num_chunks = get_virtual_pipeline_number()
        multi_chunks = get_virtual_pipeline_number() > 1

        updated, grad_norm, rollback, succeed = None, None, None, None
        it = self.states.it
        assert it == 0
        if optimizer.do_this_step:
            assert optimizer.do_prev_step
            for data_iter in conf.data_iterator:
                if data_iter is None:
                    continue
                data_iter.clear_buffer()
                data_iter.save_to_buffer()

            while it < len(conf.schedules):
                scheduled_node = conf.schedules[it]
                if multi_chunks:
                    parallel_state.set_virtual_pipeline_model_parallel_rank(scheduled_node.chunk)
                if scheduled_node.type in [FuncType.SEND_FORWARD, FuncType.RECV_FORWARD]:
                    assert scheduled_node.chunk % num_chunks == 0
                    next_is_comm = it + 1 < len(conf.schedules) and conf.schedules[it + 1].type in AUTO_SCHEDULE_COMMUNICATION_TYPES
                    next_compute = list(filter(lambda x: x.type.is_computation(), conf.schedules[it + 1:]))
                    next_compute = next_compute[0] if len(next_compute) > 0 else None
                    self.add_communication(scheduled_node, next_is_comm, next_compute)
                elif scheduled_node.type == F:
                    assert scheduled_node.chunk % num_chunks == 0
                    self.schedule_f(scheduled_node)
                elif scheduled_node.type == FuncType.RECV_POST_VALIDATION:
                    optimizer.recv_post_validation()
                elif scheduled_node.type == FuncType.SEND_POST_VALIDATION:
                    optimizer.send_post_validation()
                elif scheduled_node.type == FuncType.POST_VALIDATION:
                    self.flush()
                    updated, grad_norm, rollback, succeed = optimizer.post_validation(self._free_buffers)
                    break
                else:
                    raise ValueError(f"Unexpected type {scheduled_node.type}")
                it += 1
            assert succeed is not None
        else:
            while it < len(conf.schedules):
                scheduled_node = conf.schedules[it]
                if multi_chunks:
                    parallel_state.set_virtual_pipeline_model_parallel_rank(scheduled_node.chunk)
                if scheduled_node.type in [FuncType.SEND_FORWARD, FuncType.RECV_FORWARD, F]:
                    if optimizer.do_prev_step and scheduled_node.type == FuncType.RECV_FORWARD:
                        next_is_comm = it + 1 < len(conf.schedules) and conf.schedules[it + 1].type in AUTO_SCHEDULE_COMMUNICATION_TYPES
                        next_compute = list(filter(lambda x: x.type.is_computation(), conf.schedules[it + 1:]))
                        next_compute = next_compute[0] if len(next_compute) > 0 else None
                        self.add_communication(scheduled_node, next_is_comm, next_compute)
                elif scheduled_node.type == FuncType.RECV_POST_VALIDATION:
                    optimizer.recv_post_validation()
                elif scheduled_node.type == FuncType.SEND_POST_VALIDATION:
                    optimizer.send_post_validation()
                elif scheduled_node.type == FuncType.POST_VALIDATION:
                    self.flush()
                    updated, grad_norm, rollback, succeed = optimizer.post_validation(self._free_buffers)
                    break
                else:
                    raise ValueError(f"Unexpected type {scheduled_node.type}")
                it += 1
            assert not succeed
        if not succeed:
            if optimizer.do_prev_step:
                # send dummy recv_forward to clear send_forward request of last rank
                while it < len(conf.schedules):
                    scheduled_node = conf.schedules[it]
                    if multi_chunks:
                        parallel_state.set_virtual_pipeline_model_parallel_rank(scheduled_node.chunk)
                    if scheduled_node.type == FuncType.RECV_FORWARD and scheduled_node.rollback:
                        self.add_communication(scheduled_node, False, None)
                    it += 1
            self.reset()
            it = 0

        for data_iter in conf.data_iterator:
            if data_iter is None:
                continue
            if succeed:
                data_iter.clear_buffer()
            data_iter.pop_from_buffer()
        self.states.it = it
        return updated, grad_norm, rollback

    def schedule_f(self, scheduled_node: ScheduledNode):
        conf = self.iteration_config
        multi_chunks = get_virtual_pipeline_number() > 1
        bufs = self.buffers

        if parallel_state.is_pipeline_first_stage():
            assert len(conf.recv_tensor_shapes) > 0
            input_tensor = [None] * len(conf.recv_tensor_shapes)
        elif scheduled_node.recv_peer_stage is None or scheduled_node.recv_peer_stage == scheduled_node.stage:
            assert multi_chunks
            assert scheduled_node.chunk % 2 == 1
            assert parallel_state.is_pipeline_last_stage(ignore_virtual=True)
            input_tensor = bufs.local_send_forward_buffer[scheduled_node.seq_split_idx].pop(0)
        else:
            input_tensor, handles = bufs.recv_forward_buffer[scheduled_node.chunk][scheduled_node.seq_split_idx].pop(0)
            # Flatten for ZB
            if next(h for h in handles if isinstance(h, list)) is not None:
                handles = sum(handles, [])
            for h in handles:
                h.wait()
        assert isinstance(input_tensor, list), "input_tensor should be list of tensors"

        if get_args().profile:
            torch.cuda.nvtx.range_push(
                f'F{scheduled_node.microbatch}.{scheduled_node.chunk}.{scheduled_node.seq_split_idx}')

        if conf.run_timer:
            ScheduleTimers.for_chunk(scheduled_node.chunk).f_cnt += 1
            ScheduleTimers.for_chunk(scheduled_node.chunk).f.start()
            mem_before = torch.cuda.memory_allocated()

        parallel_state.set_seq_split_idx(scheduled_node.seq_split_idx)
        output_tensor, num_tokens = forward_step(
            conf.forward_step_func,
            conf.data_iterator[scheduled_node.chunk],
            conf.model[scheduled_node.chunk],
            conf.num_microbatches,
            input_tensor,
            bufs.forward_data_store,
            conf.config,
            conf.collect_non_loss_data,
            checkpoint_activations_microbatch=None,
        )
        assert isinstance(output_tensor, list), "output tensor should be a list"
        bufs.total_num_tokens += num_tokens.item()

        if conf.run_timer:
            ScheduleTimers.for_chunk(scheduled_node.chunk).f.stop()
            ScheduleTimers.for_chunk(scheduled_node.chunk).f_mem += torch.cuda.memory_allocated() - mem_before
        if get_args().profile:
            torch.cuda.nvtx.range_pop()

        if not parallel_state.is_pipeline_last_stage():
            if scheduled_node.send_peer_stage is None or scheduled_node.send_peer_stage == scheduled_node.stage:
                assert multi_chunks
                assert scheduled_node.chunk % 2 == 0
                assert parallel_state.is_pipeline_last_stage(ignore_virtual=True)
                detached_output_tensor = [t.detach().requires_grad_() for t in output_tensor]
                bufs.local_send_forward_buffer[scheduled_node.seq_split_idx].append(detached_output_tensor)
                deallocate_output_tensor(output_tensor[0],
                                         conf.config.deallocate_pipeline_outputs)
            else:
                bufs.send_forward_buffer[scheduled_node.chunk][scheduled_node.seq_split_idx].append(output_tensor)
        if not conf.forward_only:
            bufs.input_tensors[scheduled_node.chunk].push(input_tensor)
            bufs.output_tensors[scheduled_node.chunk].push(output_tensor)
            if parallel_state.is_pipeline_last_stage():
                deallocate_output_tensor(output_tensor[0], conf.config.deallocate_pipeline_outputs)

    def schedule_b_impl(self, scheduled_node: ScheduledNode):
        conf = self.iteration_config
        multi_chunks = get_virtual_pipeline_number() > 1
        if conf.forward_only:
            return

        bufs = self.buffers
        input_tensor = bufs.input_tensors[scheduled_node.chunk].pop()
        output_tensor = bufs.output_tensors[scheduled_node.chunk].pop()
        assert isinstance(input_tensor, list), "input_tensor should be list of tensor"
        assert isinstance(output_tensor, list), "output_tensor should be list of tensor"

        if parallel_state.is_pipeline_last_stage():
            # Keep the original behavior when we do a dummy communication
            assert len(conf.send_tensor_shapes) > 0
            output_tensor_grad = [None] * len(conf.send_tensor_shapes)
        elif scheduled_node.recv_peer_stage is None or scheduled_node.recv_peer_stage == scheduled_node.stage:
            assert multi_chunks
            assert scheduled_node.recv_peer_stage is None
            assert scheduled_node.chunk % 2 == 0
            assert parallel_state.is_pipeline_last_stage(ignore_virtual=True)
            output_tensor_grad = bufs.local_send_backward_buffer[scheduled_node.seq_split_idx].pop(0)
        else:
            output_tensor_grad, handles = bufs.recv_backward_buffer[
                scheduled_node.chunk][scheduled_node.seq_split_idx].pop(0)
            # Flatten for ZB
            if next(h for h in handles if isinstance(h, list)) is not None:
                handles = sum(handles, [])
            for h in handles:
                h.wait()
        assert isinstance(output_tensor_grad, list), "output_tensor_grad should be a list"

        if get_args().profile:
            torch.cuda.nvtx.range_push(
                f'B{scheduled_node.microbatch}.{scheduled_node.chunk}.{scheduled_node.seq_split_idx}')
        if conf.run_timer:
            ScheduleTimers.for_chunk(scheduled_node.chunk).b_cnt += 1
            ScheduleTimers.for_chunk(scheduled_node.chunk).b.start()
            mem_before = torch.cuda.memory_allocated()
        input_tensor_grad = backward_step(
            input_tensor,
            output_tensor,
            output_tensor_grad,
            conf.model_type,
            conf.config
        )
        assert isinstance(input_tensor_grad, list), "input_tensor_grad should be a list"

        if conf.run_timer:
            ScheduleTimers.for_chunk(scheduled_node.chunk).b.stop()
            ScheduleTimers.for_chunk(scheduled_node.chunk).b_mem += torch.cuda.memory_allocated() - mem_before
        if get_args().profile:
            torch.cuda.nvtx.range_pop()

        # No need to propagate gradient from the first layer.
        if not parallel_state.is_pipeline_first_stage():
            if scheduled_node.send_peer_stage is None or scheduled_node.send_peer_stage == scheduled_node.stage:
                assert multi_chunks
                assert scheduled_node.chunk % 2 == 1
                assert parallel_state.is_pipeline_last_stage(ignore_virtual=True)
                bufs.local_send_backward_buffer[scheduled_node.seq_split_idx].append(input_tensor_grad)
            else:
                bufs.send_backward_buffer[scheduled_node.chunk][scheduled_node.seq_split_idx].append(
                    input_tensor_grad)

        WeightGradStore.flush(chunk=scheduled_node.chunk, seq_split_idx=scheduled_node.seq_split_idx)

    def schedule_b(self, scheduled_node):
        with WeightGradStore.set_split_bw(True):
            self.schedule_b_impl(scheduled_node)

    def schedule_bw(self, scheduled_node):
        with WeightGradStore.set_split_bw(False):
            self.schedule_b_impl(scheduled_node)

    def schedule_w(self, scheduled_node, non_w_pending):
        conf = self.iteration_config
        multi_chunks = get_virtual_pipeline_number() > 1
        if conf.forward_only:
            return
        chunk = scheduled_node.chunk
        states = self.states

        if (not multi_chunks and non_w_pending) or \
                (multi_chunks and non_w_pending and scheduled_node.microbatch != conf.num_microbatches - 1):
            if get_args().profile:
                torch.cuda.nvtx.range_push(f'W{scheduled_node.microbatch}.{scheduled_node.chunk}.{scheduled_node.seq_split_idx}')
            if conf.run_timer:
                ScheduleTimers.for_chunk(scheduled_node.chunk).w_cnt += 1
                ScheduleTimers.for_chunk(scheduled_node.chunk).w.start()
            WeightGradStore.pop(chunk=scheduled_node.chunk, seq_split_idx=scheduled_node.seq_split_idx)
            if conf.run_timer:
                ScheduleTimers.for_chunk(scheduled_node.chunk).w.stop()
            if get_args().profile:
                torch.cuda.nvtx.range_pop()
        elif not states.w_clear_run[chunk]:
            # Clear if this is the last minibatch or there is no non-W pending
            pending_ws = WeightGradStore.queue_size(chunk, scheduled_node.seq_split_idx)
            if get_args().profile:
                torch.cuda.nvtx.range_push(f'W_clear.{chunk}.{scheduled_node.seq_split_idx}')
            if conf.run_timer:
                ScheduleTimers.for_chunk(scheduled_node.chunk).w_cnt += pending_ws
                ScheduleTimers.for_chunk(scheduled_node.chunk).w.start()
            WeightGradStore.clear(conf.model[chunk], chunk=chunk, seq_split_idx=scheduled_node.seq_split_idx)
            if conf.run_timer:
                ScheduleTimers.for_chunk(scheduled_node.chunk).w.stop()
            if get_args().profile:
                torch.cuda.nvtx.range_pop()  # W
            states.w_clear_run[chunk] = True

    def add_communication(
        self,
        scheduled_node: ScheduledNode,
        next_is_comm: bool,
        next_compute: Optional[ScheduledNode],
    ):
        conf = self.iteration_config
        states = self.states

        if conf.forward_only and scheduled_node.type.is_backward():
            return
        states.communication_batch[self.direction_map(scheduled_node)].append(
            (scheduled_node, conf.tensor_shape))
        def is_consumer(scheduled_node, next_compute):
            if scheduled_node.chunk == next_compute.chunk \
                    and scheduled_node.seq_split_idx == next_compute.seq_split_idx \
                    and scheduled_node.microbatch == next_compute.microbatch:
                if scheduled_node.type == FuncType.RECV_FORWARD and next_compute.type == F:
                    return True
                if scheduled_node.type == FuncType.RECV_BACKWARD and next_compute.type in (B, BW):
                    return True
            return False
        if (next_compute is not None and is_consumer(scheduled_node, next_compute)) or not next_is_comm or conf.forward_only:
            self.flush()

    def flush(self):
        conf = self.iteration_config
        states = self.states
        bufs = self.buffers

        name = '_'.join(
            [f'{v[0].type}.{v[0].microbatch}.{v[0].chunk}.{v[0].seq_split_idx}' for v in itertools.chain(*[vs for k, vs in states.communication_batch.items()])])

        assert conf.send_tensor_shapes == conf.recv_tensor_shapes
        assert len(conf.send_tensor_shapes) == 1
        assert conf.send_tensor_shapes[0] == conf.tensor_shape

        sn_tensors = [
            bufs.buffer_map(x[0]).pop(0)[0]
            for x in states.communication_batch['SEND_NEXT']
        ]
        sp_tensors = [
            bufs.buffer_map(x[0]).pop(0)[0]
            for x in states.communication_batch['SEND_PREV']
        ]
        rn_tensors = [
            torch.empty(
                conf.tensor_shape,
                requires_grad=True,
                device=torch.cuda.current_device(),
                dtype=conf.config.pipeline_dtype,
            ) for x in states.communication_batch['RECV_NEXT']
        ]
        assert conf.recv_tensor_shapes[0] == conf.tensor_shape
        rp_tensors = [
            torch.empty(
                conf.tensor_shape,
                requires_grad=True,
                device=torch.cuda.current_device(),
                dtype=conf.config.pipeline_dtype,
            ) for x in states.communication_batch['RECV_PREV']
        ]
        if get_args().profile:
            torch.cuda.nvtx.range_push(name)
        req = fused_pipeline_ops(
            sp_tensors,
            rp_tensors,
            sn_tensors,
            rn_tensors,
        )
        if get_args().profile:
            torch.cuda.nvtx.range_pop()
        # We don't care about the reqs order here, all users need to all reqs to finish
        for x in states.communication_batch['RECV_NEXT']:
            bufs.buffer_map(x[0]).append(([rn_tensors.pop(0)], [req]))
        for x in states.communication_batch['RECV_PREV']:
            bufs.buffer_map(x[0]).append(([rp_tensors.pop(0)], [req]))
        states.send_handles.append([req])
        assert(not rn_tensors)
        assert(not rp_tensors)
        for direction in ['SEND_PREV', 'SEND_NEXT']:
            for idx, x in enumerate(states.communication_batch[direction]):
                if x[0].type == FuncType.SEND_FORWARD:
                    deallocate_output_tensor(sp_tensors[idx] if direction == 'SEND_PREV' else sn_tensors[idx],
                                             conf.config.deallocate_pipeline_outputs)
        for k, v in states.communication_batch.items():
            v.clear()

    def wait_for_comm(self):
        for handles in self.states.send_handles:
            # Flatten handles for ZB
            if next(h for h in handles if isinstance(h, list)) is not None:
                handles = sum(handles, [])
            for h in handles:
                h.wait()

    @classmethod
    def direction_map(cls, node):
        sr = "SEND_" if node.type.is_send() else "RECV_"
        d = "NEXT" if node.comm_direction == CommDirection.NEXT else "PREV"
        direction = sr + d
        return direction

    def disable_grad_sync(self):
        """Disable asynchronous grad reductions"""
        if self.no_sync_context is None:
            self.no_sync_context = self.iteration_config.no_sync_func()
            self.no_sync_context.__enter__()

    def enable_grad_sync(self):
        """Enable asynchronous grad reductions"""
        if self.no_sync_context is not None:
            self.no_sync_context.__exit__(None, None, None)
            self.no_sync_context = None


class SchedNodeRuntime:
    def __init__(self):
        self.no_sync_context = None
        self.no_sync_func = None

        self.iteration_id = 0

        self.curr_iteration: Optional[TrainingIteration] = None
        self.next_iteration: Optional[TrainingIteration] = None

    def gen_it_id(self):
        self.iteration_id += 1
        return self.iteration_id - 1

    def prepare(
        self,
        schedule: List[ScheduledNode],
        forward_step_func,
        data_iterator: Union[Iterator, List[Iterator]],
        model: Union[torch.nn.Module, List[torch.nn.Module]],
        num_microbatches: int,
        seq_length: int,
        micro_batch_size: int,
        decoder_seq_length: int = None,
        forward_only: bool = False,
        collect_non_loss_data: bool = False,
    ):
        if not isinstance(model, list):
            model = [model]
        assert len(model) > 0, "empty model list found"
        assert all(isinstance(chunk, torch.nn.Module) for chunk in model), "invalid model chunking"
        # assert data_iterator is not None, "None data_iterator found"
        if not isinstance(data_iterator, list):
            data_iterator = [data_iterator]
        assert len(data_iterator) > 0, "empty data_iterator list found"
        config = get_model_config(model[0])

        multi_chunks = get_virtual_pipeline_number() > 1
        if not multi_chunks:
            if config.overlap_p2p_comm:
                raise ValueError(
                    "Non-interleaved pipeline parallelism does not support overlapping p2p communication"
                )
        elif config.overlap_p2p_comm and config.batch_p2p_comm:
            raise ValueError("Can not use both overlap_p2p_comm and batch_p2p_comm")

        # Disable async grad reductions
        no_sync_func = config.no_sync_func
        if isinstance(no_sync_func, list):

            def multi_no_sync():
                stack = contextlib.ExitStack()
                for model_chunk_no_sync_func in config.no_sync_func:
                    stack.enter_context(model_chunk_no_sync_func())
                return stack

            no_sync_func = multi_no_sync
        # no_sync_func is not supported now.
        assert no_sync_func is None, "Sync func is not supported yet"
        if no_sync_func is None:
            no_sync_func = contextlib.nullcontext
        self.no_sync_func = no_sync_func
        self.no_sync_context = None

        assert config.param_sync_func is None, "Param sync func is not supported yet"

        # Checkpoint the activations of partial Transformer layers in a number of micro-batches
        # within the maximum outstanding micro-batch backpropagations.
        # Micro-batches with the ids less than 'num_microbatches_with_partial_activation_checkpoints'
        # checkpoint partial Transformer layers (or skip checkpointing) and
        # the rest of micro-batches within a window of micro-batches checkpoint
        # all Transformer layers. The window of micro-batches is set by the maximum
        # outstanding backpropagations and becomes smaller at later pipeline stages.
        # Please refer the appendix C in https://arxiv.org/pdf/2205.05198.pdf
        assert config.num_microbatches_with_partial_activation_checkpoints is None

        model_type = get_model_type(model[0])
        encoder_decoder_xattn = get_model_xattn(model[0])

        tensor_shape = [seq_length, micro_batch_size, config.hidden_size]
        if config.sequence_parallel:
            tensor_shape[0] = (
                tensor_shape[0] // parallel_state.get_tensor_model_parallel_world_size()
            )
        tensor_shape = tuple(tensor_shape)

        if multi_chunks and decoder_seq_length is not None and decoder_seq_length != tensor_shape[0]:
            raise RuntimeError(
                "Interleaving is not supported with a different decoder sequence length."
            )

        rank = parallel_state.get_pipeline_model_parallel_rank()
        recv_tensor_shapes = get_tensor_shapes(
            rank=rank - 1,
            model_type=model_type,
            seq_length=seq_length,
            micro_batch_size=micro_batch_size,
            decoder_seq_length=decoder_seq_length,
            config=config,
            encoder_decoder_xattn=encoder_decoder_xattn,
        )
        assert recv_tensor_shapes[0] == tensor_shape
        send_tensor_shapes = get_tensor_shapes(
            rank=rank,
            model_type=model_type,
            seq_length=seq_length,
            micro_batch_size=micro_batch_size,
            decoder_seq_length=decoder_seq_length,
            config=config,
            encoder_decoder_xattn=encoder_decoder_xattn,
        )
        assert send_tensor_shapes[0] == tensor_shape

        if not forward_only:
            ScheduleTimers.iter_counter += 1
        run_timer = (
            get_args().zero_bubble_pipeline_timers_end_iter
            >= ScheduleTimers.iter_counter
            >= get_args().zero_bubble_pipeline_timers_start_iter
        )

        bootstrap_and_profile_p2p_communication(config, [tensor_shape], [tensor_shape])

        iteration_config = TrainingIterationConfig(
            run_timer=run_timer,
            schedules=schedule,
            forward_step_func=forward_step_func,
            data_iterator=data_iterator,
            model=model,
            model_type=model_type,
            config=config,
            num_microbatches=num_microbatches,
            forward_only=forward_only,
            collect_non_loss_data=collect_non_loss_data,
            no_sync_func=no_sync_func,
            tensor_shape=tensor_shape,
            recv_tensor_shapes=recv_tensor_shapes,
            send_tensor_shapes=send_tensor_shapes,
        )
        return iteration_config

    def run(self, *args, **kwargs):
        # 3 cases that need to initialize current_iteration:
        # - First training iteration
        #     When post validation is enabled, the curr_iteration is initialized
        #     in the last loop during running optimizer.
        #     But if this is the very first iteration, there's no last loop.
        #     So need to initialize.
        # - Post validation is disabled
        #     This could be disabled by config or
        #     optimizer.post_validation_enabled is False because optimizer is not ready yet.
        #     No initialization is done in optimizer step. So need to init.
        # - Forward-only mode
        #     To training so optimizer. Similar as above.
        if self.curr_iteration is None or \
                not get_args().enable_optimizer_post_validation or \
                self.next_iteration is None or \
                kwargs['forward_only']:
            iteration_config = self.prepare(*args, **kwargs)
            self.curr_iteration = TrainingIteration(iteration_config, self.gen_it_id())
        else:
            assert self.next_iteration
            self.curr_iteration = self.next_iteration
            self.next_iteration = None

        result = self.curr_iteration.run()
        return result

    def post_validate(self, optimizer, *args, **kwargs):
        iteration_config = self.prepare(*args, **kwargs)
        self.curr_iteration.reset()  # Explicitly free memory
        self.next_iteration = TrainingIteration(iteration_config, self.gen_it_id())
        # Next iteration will be responsible for the post validation of current iteration
        return self.next_iteration.run_until_post_validation(optimizer)

    def __call__(self, *args, **kwargs):
        optimizer = kwargs.get("optimizer")
        if "optimizer" in kwargs:
            kwargs.pop("optimizer")
        if optimizer is None:
            result = self.run(*args, **kwargs)
        else:
            result = self.post_validate(optimizer, *args, **kwargs)
        return result


def get_virtual_pipeline_number():
    return parallel_state.get_virtual_pipeline_model_parallel_world_size() or 1


def fused_pipeline_ops(
    tensor_send_prev: List[torch.Tensor],
    tensor_recv_prev: List[torch.Tensor],
    tensor_send_next: List[torch.Tensor],
    tensor_recv_next: List[torch.Tensor],
):
    ops = []
    group = get_pipeline_model_parallel_group()
    for t in tensor_send_prev:
        send_prev_op = torch.distributed.P2POp(
            torch.distributed.isend,
            t,
            get_pipeline_model_parallel_prev_rank(),
            group,
        )
        ops.append(send_prev_op)
    for t in tensor_recv_prev:
        recv_prev_op = torch.distributed.P2POp(
            torch.distributed.irecv,
            t,
            get_pipeline_model_parallel_prev_rank(),
            group,
        )
        ops.append(recv_prev_op)
    for t in tensor_send_next:
        send_next_op = torch.distributed.P2POp(
            torch.distributed.isend,
            t,
            get_pipeline_model_parallel_next_rank(),
            group,
        )
        ops.append(send_next_op)
    for t in tensor_recv_next:
        recv_next_op = torch.distributed.P2POp(
            torch.distributed.irecv,
            t,
            get_pipeline_model_parallel_next_rank(),
            group,
        )
        ops.append(recv_next_op)
    if len(ops) > 0:
        reqs = torch.distributed.batch_isend_irecv(ops)
    else:
        reqs = []
    return reqs


def bootstrap_and_profile_p2p_communication(config, send_tensor_shapes, recv_tensor_shapes):
    # When we fuse some send-recv communication ops in a device and can't fuse on other devices
    # because there are computation between communication, it will result in deadlock.
    # Doing send-recv without fusing using the same communicator beforehand can avoid this problem.
    # Pytorch internally can possibly use different communicator for send-recv:
    #    (1) send recv without batch_isend_irecv use a communicator for each specific send-recv device pair.
    #    (2) send recv inside a batch_isend_irecv use global (collective) communicator.
    # Related codes are in ProcessGroupNCCL::pointToPoint()
    # where different formats of communicator key are uses.
    # Related post: https://github.com/pytorch/pytorch/issues/129140
    # To ensure we use the same communicator here and the communication later,
    # we enforce using global communicator by calling batch_isend_irecv.
    # For this, batch_p2p_comm should be True and overlap_p2p_comm should be False.
    # --no-overlap-p2p-communication must be specified.
    assert config.batch_p2p_comm, "--no-overlap-p2p-communication must be used for zero-bubble runtime."
    if (
        ScheduleTimers.iter_counter == 1
        and parallel_state.get_pipeline_model_parallel_world_size() > 1
    ):
        nccl_init_tensor = [torch.Tensor([0]).cuda()]
        shape = [(1,)]
        if get_args().zero_bubble_v_schedule or get_args().enable_1f1b_v:
            # Make everyone think they are the first chunk, so we still need additional check to prevent rank -1 to send_forward/recv_backward
            parallel_state.set_virtual_pipeline_model_parallel_rank(0)
        if not parallel_state.is_pipeline_first_stage(ignore_virtual=True):
            recv_forward(shape, config)
        if not parallel_state.is_pipeline_last_stage(ignore_virtual=True):
            send_forward(nccl_init_tensor, shape, config)
            recv_backward(shape, config)
        if not parallel_state.is_pipeline_first_stage(ignore_virtual=True):
            send_backward(nccl_init_tensor, shape, config)
        # for interleaved pipeline parallelism
        if parallel_state.is_pipeline_first_stage(ignore_virtual=True):
            _communicate(
                tensor_send_next=None,
                tensor_send_prev=None,
                recv_prev=True,
                recv_next=False,
                tensor_shape=shape[0],
                config=config,
            )
            _communicate(
                tensor_send_next=None,
                tensor_send_prev=nccl_init_tensor[0],
                recv_prev=False,
                recv_next=False,
                tensor_shape=None,
                config=config,
            )
        if parallel_state.is_pipeline_last_stage(ignore_virtual=True):
            _communicate(
                tensor_send_next=nccl_init_tensor[0],
                tensor_send_prev=None,
                recv_prev=False,
                recv_next=False,
                tensor_shape=None,
                config=config,
            )
            _communicate(
                tensor_send_next=None,
                tensor_send_prev=None,
                recv_prev=False,
                recv_next=True,
                tensor_shape=shape[0],
                config=config,
            )

        # Benchmarking the communication cost
        send_data = [
            torch.zeros(*shape, dtype=config.pipeline_dtype).cuda() for shape in send_tensor_shapes
        ]
        recv_data = [
            torch.zeros(*shape, dtype=config.pipeline_dtype).cuda() for shape in recv_tensor_shapes
        ]
        torch.distributed.barrier()
        t = Timer('comm-benchmark')
        t.start()
        print_rank_0(
            f"Start benchmarking communication with size {recv_tensor_shapes}, {send_tensor_shapes}"
        )
        for _ in range(10):
            if not parallel_state.is_pipeline_first_stage(ignore_virtual=True):
                recv_forward(recv_tensor_shapes, config)
            if not parallel_state.is_pipeline_last_stage(ignore_virtual=True):
                send_forward(send_data, send_tensor_shapes, config)
                recv_backward(send_tensor_shapes, config)
            if not parallel_state.is_pipeline_first_stage(ignore_virtual=True):
                send_backward(recv_data, recv_tensor_shapes, config)
        t.stop()
        per_communication = torch.cuda.FloatTensor(
            [t.elapsed() / (parallel_state.get_pipeline_model_parallel_world_size() - 1) / 2 / 10]
        )
        torch.distributed.all_reduce(per_communication, torch.distributed.ReduceOp.MAX)
        ScheduleTimers.comm_time = per_communication.item()
        print_rank_0(f"Communication time: {ScheduleTimers.comm_time}")


shed_node_runtime = SchedNodeRuntime()


def get_zb_runtime_instance():
    return shed_node_runtime


schedule_cache = None
is_auto_schedule = False


def update_schedule(scheduler, f: List[int], b: List[int], w: List[int],
                    c: int, f_mem: List[int], b_mem: List[int], w_mem: List[int],
                    mem_limit: int):
    pipeline_model_parallel_size = parallel_state.get_pipeline_model_parallel_world_size()
    ag_arguments = [None] * torch.distributed.get_world_size()
    torch.distributed.all_gather_object(ag_arguments, (f, b, w, f_mem, b_mem, w_mem, mem_limit))
    assert len(ag_arguments) == torch.distributed.get_world_size()
    # Each value is an array of dimension (device, chunk)
    f, b, w, f_mem, b_mem, w_mem, mem_limit = zip(*ag_arguments)

    if is_second_last_pipeline_stage():
        print(
            f"rank {torch.distributed.get_rank()} Performing ILP with: f={f},\n b={b},\n w={w},\n c={c},\n f_mem={f_mem},\n b_mem={b_mem},\n w_mem={w_mem},\n mem_limit={mem_limit}")
        global schedule_cache
        schedule_cache = scheduler(
            pipeline_model_parallel_size,
            get_num_microbatches(),
            f, b, w,
            max(c, 1),
            f_mem, b_mem, w_mem,
            mem_limit,
        )
        ag_result = [None] * torch.distributed.get_world_size()
        torch.distributed.all_gather_object(ag_result, schedule_cache)

    else:
        ag_result = [None] * torch.distributed.get_world_size()
        torch.distributed.all_gather_object(ag_result, None)
        schedule_cache = list(filter(lambda x: x is not None, ag_result))
        assert len(schedule_cache) == 1
        schedule_cache = schedule_cache[0]
    return schedule_cache


def get_zero_bubble_forward_backward_func():
    pipeline_model_parallel_size = parallel_state.get_pipeline_model_parallel_world_size()
    assert pipeline_model_parallel_size > 1, "zero bubble must be enabled with pipeline parallelism"

    args = get_args()
    hidden_size = args.hidden_size
    num_attention_heads = args.num_attention_heads
    seq_length = args.seq_length
    f_mem_approx = 34 * hidden_size + 5 * num_attention_heads * seq_length
    w_mem_approx = - 32 * hidden_size
    b_mem_approx = - f_mem_approx - w_mem_approx

    def wrapped_auto_schedule_forward_backward_func(func, scheduler):
        global schedule_cache, is_auto_schedule
        if schedule_cache is None:
            schedule_cache = update_schedule(scheduler,
                                             f=[1000],
                                             b=[1000],
                                             w=[1000],
                                             c=1,
                                             f_mem=[f_mem_approx],
                                             b_mem=[0],
                                             w_mem=[-f_mem_approx],
                                             mem_limit=f_mem_approx * parallel_state.get_pipeline_model_parallel_world_size())
            # Using fixed 1p schedule
        if ScheduleTimers.concluded and not is_auto_schedule:
            conclusion = ScheduleTimers.joint_conclusion()
            # TODO(wanxy): Maybe an all-reduce here to collect global stats?
            print(f"rank {torch.distributed.get_rank()} profiling conclusion: {conclusion}")

            def estimate_free_memory_on_this_rank(old_schedule):
                (memory_free, memory_all) = [x // 1000000 for x in torch.cuda.mem_get_info()]
                memory_all = memory_all * get_args().zero_bubble_adaptive_memory_limit_percentile / 100
                activation_cost = 0
                stage = parallel_state.get_pipeline_model_parallel_rank()
                max_activation = 0
                for node in old_schedule[stage]:
                    chunk = node.chunk if hasattr(node, 'chunk') else 0
                    if node.type == F:
                        activation_cost += conclusion[4][chunk]
                    elif node.type == B:
                        activation_cost += conclusion[5][chunk]
                    elif node.type == W:
                        activation_cost += conclusion[6][chunk]
                    elif node.type == BW:
                        activation_cost += conclusion[5][chunk]
                        activation_cost += conclusion[6][chunk]
                    max_activation = max(activation_cost, max_activation)
                free_mem = memory_all - (torch.cuda.max_memory_allocated() // 1000000 - max_activation)

                print(f'estimated max free memory for activations on rank {torch.distributed.get_rank()} \
                    memory_free: {memory_free}, memory_all: {memory_all}, max_activation: {max_activation}, \
                    max_allocated: {torch.cuda.max_memory_allocated() // 1000000}, \
                    current_allocated: {torch.cuda.memory_allocated() // 1000000}, \
                    free_mem: {free_mem}')

                print(f'rank {torch.distributed.get_rank()} mem summary {torch.cuda.memory_summary()}')
                return free_mem

            schedule_cache = update_schedule(scheduler,
                                             *conclusion,
                                             mem_limit=estimate_free_memory_on_this_rank(schedule_cache))
            is_auto_schedule = True

        def wrap_schedule(**kwargs):
            # print(f"DEBUG wrap_schedule data_iterator {kwargs.get('data_iterator')}")
            # assert kwargs.get('data_iterator') is not None, "data_iterator found none in wrap_schedule"
            return func(
                schedule=schedule_cache[parallel_state.get_pipeline_model_parallel_rank()], **kwargs
            )

        return wrap_schedule

    def avg_then_mid(a: List[List[float]]):
        a = [sum(x) / len(x) for x in a]
        return max(sorted(a)[len(a) // 2], 1)

    if get_args().num_seq_splits > 1:
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
                max_chunks=1,
            )
            print(f"using seq 1f1b")
            local_order = seq1f1b.create_schedule(config)
            ret = run_schedule_passes(config, local_order)
            return ret

        global_zb_runtime = get_zb_runtime_instance()
        forward_backward_func = wrapped_auto_schedule_forward_backward_func(global_zb_runtime, scheduler=scheduler)
        return forward_backward_func

    if get_args().enable_1f1b_v:
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
                max_chunks=2,
            )
            local_order = v1f1b.create_schedule(config)
            ret = run_schedule_passes(config, local_order)
            return ret

        global_zb_runtime = get_zb_runtime_instance()
        forward_backward_func = wrapped_auto_schedule_forward_backward_func(global_zb_runtime, scheduler=scheduler)
        return forward_backward_func

    # Interleaved pipeline
    if not get_args().zero_bubble_v_schedule and not get_args().enable_zero_bubble \
            and parallel_state.get_virtual_pipeline_model_parallel_world_size() is not None \
            and parallel_state.get_virtual_pipeline_model_parallel_world_size() > 1:
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
                max_chunks=parallel_state.get_virtual_pipeline_model_parallel_world_size(),
            )
            print(f"using interleaved 1f1b")
            local_order = vpp.create_schedule(config)
            ret = run_schedule_passes(config, local_order)
            return ret

        global_zb_runtime = get_zb_runtime_instance()
        forward_backward_func = wrapped_auto_schedule_forward_backward_func(global_zb_runtime, scheduler=scheduler)
        return forward_backward_func

    if parallel_state.get_virtual_pipeline_model_parallel_world_size() is not None:
        def scheduler(nstages, nmb, f, b, w, c, _f_mem, _b_mem, _w_mem, _mem_limit):
            # For V schedule, we take average on each stage and then use mid value cross each stage.
            f_mid = avg_then_mid(f)
            b_mid = avg_then_mid(b)
            w_mid = avg_then_mid(w)
            if get_args().zero_bubble_v_schedule_mem_setup != 'zb':
                config = zb.GraphConfig(
                    cost_f=[1000.0 for _ in range(nstages)],
                    cost_b=[1000.0 for _ in range(nstages)],
                    cost_w=[1000.0 for _ in range(nstages)],
                    cost_comm=1.0,
                    mem_f=[f_mem_approx for _ in range(nstages)],
                    mem_b=[b_mem_approx for _ in range(nstages)],
                    mem_w=[w_mem_approx for _ in range(nstages)],
                    max_mem=None,
                    print_scaling=1000,
                    max_chunks=2,
                    n_stages=nstages,
                    n_micro=nmb,
                )
                # Use fixed schedule for now
                pp_graph = zbv_greedy.PipelineGraph(
                    nstages, nmb, get_args().zero_bubble_v_schedule_mem_setup, int(1000), int(1000), int(1000), int(1)
                )
                local_order = pp_graph.create_schedule(config)
                ret = run_schedule_passes(config, local_order, validate=False)
                return ret
            config = zb.GraphConfig(
                cost_f=[float(f_mid) for _ in range(nstages)],
                cost_b=[float(b_mid) for _ in range(nstages)],
                cost_w=[float(w_mid) for _ in range(nstages)],
                cost_comm=float(c),
                mem_f=[f_mem_approx for _ in range(nstages)],
                mem_b=[b_mem_approx for _ in range(nstages)],
                mem_w=[w_mem_approx for _ in range(nstages)],
                max_mem=None,
                print_scaling=1000,
                max_chunks=2,
                n_stages=nstages,
                n_micro=nmb,
            )
            pp_graph = zbv.PipelineGraph(
                nstages,
                nmb,
                f_mid, b_mid, w_mid, c,
                # V schedule does not consider memory differences between stages for now.
                f_mem=f_mem_approx, b_mem=b_mem_approx, w_mem=w_mem_approx,
                max_mem=None
                # Mem ignored for now
            )
            local_order = pp_graph.create_schedule(config)
            ret = run_schedule_passes(config, local_order, validate=False)
            return ret

        if get_args().zero_bubble_v_schedule:
            global_zb_runtime = get_zb_runtime_instance()
            forward_backward_func = wrapped_auto_schedule_forward_backward_func(global_zb_runtime,
                                                                                scheduler=scheduler)
            # forward_backward_func = wrapped_auto_schedule_forward_backward_func(forward_backward_pipelining_with_interleaving_auto_schedule,
            #                                                                     scheduler=scheduler)
        else:
            raise ValueError("got virtual pipeline parallel but v_schedule is disabled")
    else:
        def scheduler(nstages, nmb, f, b, w, c, f_mem, b_mem, w_mem, mem_limit):
            f = [x[0] for x in f]
            b = [x[0] for x in b]
            w = [x[0] for x in w]
            # Using uniform f/b/w timing for now.
            f = [sorted(f)[len(f) // 2]] * len(f)
            b = [sorted(b)[len(b) // 2]] * len(b)
            w = [sorted(w)[len(w) // 2]] * len(w)
            f_mem = [x[0] for x in f_mem]
            b_mem = [x[0] for x in b_mem]
            w_mem = [x[0] for x in w_mem]

            if args.zero_bubble_max_pending_backward != 'auto':
                print(f'manual mem limit: {args.zero_bubble_max_pending_backward * max(f_mem[:2])}')
                mem_limit = [args.zero_bubble_max_pending_backward * max(f_mem[:2])] * len(f_mem)
            else:
                print(f'adaptive mem limit: {mem_limit}')

            config = zb.GraphConfig(
                cost_f=list(map(float, f)),
                cost_b=list(map(float, b)),
                cost_w=list(map(float, w)),
                cost_comm=float(c),
                mem_f=f_mem,
                mem_b=b_mem,
                mem_w=w_mem,
                max_mem=mem_limit,
                print_scaling=1000,
                n_stages=nstages,
                n_micro=nmb,
            )
            local_order = zb.create_schedule(config)
            ret = run_schedule_passes(config, local_order, validate=False)
            return ret

        global_zb_runtime = get_zb_runtime_instance()
        forward_backward_func = wrapped_auto_schedule_forward_backward_func(global_zb_runtime, scheduler=scheduler)

    return forward_backward_func
