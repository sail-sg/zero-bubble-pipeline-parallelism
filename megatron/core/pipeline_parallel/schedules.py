# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

import contextlib
from typing import Callable, Iterator, List, Optional, Union

import torch
from torch.autograd.variable import Variable

from megatron.training import get_args
from megatron.core import parallel_state
from megatron.core.enums import ModelType
from megatron.core.pipeline_parallel import p2p_communication
from megatron.core.pipeline_parallel.schedule_timers import ScheduleTimers
from megatron.core.tensor_parallel.cross_entropy_store import CrossEntropyStore
from megatron.core.tensor_parallel.embedding_store import EmbeddingStore
from megatron.core.timers import Timer
from megatron.core.transformer.moe.router import MoEAuxLossAutoScaler
from megatron.core.utils import (
    drain_embedding_wgrad_compute,
    get_attr_wrapped_model,
    get_model_config,
    get_model_type,
    get_model_xattn,
    reset_random_state,
)
from megatron.training import get_args

# Types
Shape = Union[List[int], torch.Size]


def get_forward_backward_func():
    """Retrieves the appropriate forward_backward function given the
    configuration of parallel_state.

    Returns a function that will perform all of the forward and
    backward passes of the model given the pipeline model parallel
    world size and virtual pipeline model parallel world size in the
    global parallel_state.

    Note that if using sequence parallelism, the sequence length component of
    the tensor shape is updated to original_sequence_length /
    tensor_model_parallel_world_size.

    The function returned takes the following arguments:

    forward_step_func (required): A function that takes a data
        iterator and a model as its arguments and return the model's
        forward output and the loss function. The loss function should
        take one torch.Tensor and return a torch.Tensor of loss and a
        dictionary of string -> torch.Tensor.

        A third argument, checkpoint_activations_microbatch, indicates
        that the activations for this microbatch should be
        checkpointed. A None value for this argument indicates that
        the default from the configuration should be used. This is
        used when the
        num_microbatches_with_partial_activation_checkpoints is used.

        For example:

        def loss_func(loss_mask, output_tensor):
            losses = output_tensor.float()
            loss_mask = loss_mask.view(-1).float()
            loss = torch.sum(losses.view(-1) * loss_mask) / loss_mask.sum()

            # Reduce loss for logging.
            averaged_loss = average_losses_across_data_parallel_group([loss])

            return loss, {'lm loss': averaged_loss[0]}

        def forward_step(data_iterator, model):
            data, loss_mask = next(data_iterator)
            output = model(data)
            return output, partial(loss_func, loss_mask)


        forward_backward_func(forward_step_func=forward_step, ...)


    data_iterator (required): an iterator over the data, will be
        passed as is to forward_step_func. Expected to be a list of
        iterators in the case of interleaved pipeline parallelism.

    model (required): the actual model. Expected to be a list of modules in the case of interleaved
        pipeline parallelism. Must be a (potentially wrapped) megatron.core.models.MegatronModule.

    num_microbatches (int, required):
        The number of microbatches to go through

    seq_length (int, required): Sequence length of the current global batch. If this is a dual-stack
        transformer, this is the encoder's sequence length. This is ignored if variable_seq_lengths
        in the config is True. Otherwise, each microbatch in the current global batch size must use
        this sequence length.

    micro_batch_size (int, required): The number of sequences in a microbatch.

    decoder_seq_length (int, optional): The sequence length for the decoder in a dual-stack
        transformer. This is ignored for a single-stack transformer.

    forward_only (optional, default = False): Perform only the forward step

    collect_non_loss_data (optional, bool, default=False): TODO

    first_val_step (bool, optional): Is the first step of the validation phase. Used by
        Transformer Engine modules to only update their fp8 weights only on the first validation step.

    """
    if get_args().enable_zero_bubble or get_args().enable_1f1b_v or get_args().num_seq_splits > 1 or get_args().enable_zb_runtime:
        from megatron.core.pipeline_parallel.zerobubble import get_zero_bubble_forward_backward_func
        return get_zero_bubble_forward_backward_func()
    pipeline_model_parallel_size = parallel_state.get_pipeline_model_parallel_world_size()
    if pipeline_model_parallel_size > 1:
        if parallel_state.get_virtual_pipeline_model_parallel_world_size() is not None:
            forward_backward_func = forward_backward_pipelining_with_interleaving
        else:
            if get_args().enable_vocab_parallel:
                forward_backward_func = forward_backward_pipelining_with_split_vocab_parallel
            else:
                forward_backward_func = forward_backward_pipelining_without_interleaving
    else:
        forward_backward_func = forward_backward_no_pipelining
    return forward_backward_func


def deallocate_output_tensor(out, deallocate_pipeline_outputs=False):
    '''Pseudo-deallocate (i.e., set to scalar) the output tensor's '.data' field.

    This method should be called right after the output tensor has been
    sent to the next pipeline stage. At this point, the output tensor is
    only useful for its '.grad_fn' field, and not its '.data'.
    '''
    if (out is None) or (not deallocate_pipeline_outputs):
        return
    assert isinstance(out, torch.Tensor), "expected Tensor, found %s." % type(out).__name__
    assert out._base is None, "counter-productive to free a view of another tensor."
    out.data = torch.empty(
        (1,),
        device=out.device,
        dtype=out.dtype,
    )


def custom_backward(output, grad_output):
    '''Directly call C++ autograd engine.

    To make the 'deallocate_output_tensor' (above) optimization work, the C++
    autograd engine must be called directly, bypassing Pytorch's
    torch.autograd.backward. Pytorch's 'backward' checks that the output and
    grad have the same shape, while C++'s 'backward' does not.
    '''

    assert output.numel() == 1, "output should be pseudo-'freed' in schedule, to optimize memory"
    assert isinstance(output, torch.Tensor), "output == '%s'." % type(output).__name__
    assert isinstance(grad_output, (torch.Tensor, type(None))), (
        "grad_output == '%s'." % type(grad_output).__name__
    )

    # Handle scalar output
    if grad_output is None:
        assert output.numel() == 1, "implicit grad requires scalar output."
        grad_output = torch.ones_like(
            output,
            memory_format=torch.preserve_format,
        )

    # Call c++ engine [ see torch/csrc/autograd/python_engine.cpp ]
    Variable._execution_engine.run_backward(
        tensors=(output,),
        grad_tensors=(grad_output,),
        keep_graph=False,
        create_graph=False,
        inputs=tuple(),
        allow_unreachable=True,
        accumulate_grad=True,
    )


def set_current_microbatch(model, microbatch_id):
    decoder_exists = True
    decoder = None
    try:
        decoder = get_attr_wrapped_model(model, "decoder")
    except RuntimeError:
        decoder_exists = False
    if decoder_exists and decoder is not None:
        decoder.current_microbatch = microbatch_id


def forward_step(
    forward_step_func,
    data_iterator,
    model,
    num_microbatches,
    input_tensor,
    forward_data_store,
    config,
    collect_non_loss_data=False,
    checkpoint_activations_microbatch=None,
    is_first_microbatch=False,
    current_microbatch=None,
    encoder_decoder_xattn=False,
    skip_loss_compute=False,
    force_loss_compute=False,
    run_timer=False,
):
    """Forward step for passed-in model.

    If first stage, input tensor is obtained from data_iterator, otherwise
    passed-in input_tensor is used.

    Returns output tensor."""
    if config.timers is not None:
        config.timers('forward-compute', log_level=2).start()
    if get_args().enable_exactly_numeric_match:
        reset_random_state()

    if run_timer:
        ScheduleTimers.for_chunk(0).f_cnt += 1
        ScheduleTimers.for_chunk(0).f.start()
    mem_before = torch.cuda.memory_allocated()

    if is_first_microbatch and hasattr(model, 'set_is_first_microbatch'):
        model.set_is_first_microbatch()
    if current_microbatch is not None:
        set_current_microbatch(model, current_microbatch)

    unwrap_output_tensor = False
    if not isinstance(input_tensor, list):
        input_tensor = [input_tensor]
        unwrap_output_tensor = True

    set_input_tensor = get_attr_wrapped_model(model, "set_input_tensor")
    set_input_tensor(input_tensor)

    if config.enable_autocast:
        context_manager = torch.autocast("cuda", dtype=config.autocast_dtype)
    else:
        context_manager = contextlib.nullcontext()
    with context_manager:
        if checkpoint_activations_microbatch is None:
            if get_args().profile:
                torch.cuda.nvtx.range_push('forward_step_func')
            output_tensor, loss_func = forward_step_func(data_iterator, model,
                                                         microbatch_id=current_microbatch)
            if get_args().profile:
                torch.cuda.nvtx.range_pop()
        else:
            output_tensor, loss_func = forward_step_func(
                data_iterator, model, checkpoint_activations_microbatch,
                microbatch_id=current_microbatch,
            )

    num_tokens = torch.tensor(0, dtype=torch.int)
    if ((parallel_state.is_pipeline_last_stage()) and (not skip_loss_compute)) or (force_loss_compute):
        if get_args().enable_vocab_parallel:
            output_tensor = output_tensor.transpose(0, 1).contiguous()
        if not collect_non_loss_data:
            outputs = loss_func(output_tensor)
            if len(outputs) == 3:
                output_tensor, num_tokens, loss_reduced = outputs
                if not config.calculate_per_token_loss:
                    output_tensor /= num_tokens
                    output_tensor /= num_microbatches
            else:
                # preserve legacy loss averaging behavior (ie, over the number of microbatches)
                assert len(outputs) == 2
                output_tensor, loss_reduced = outputs
                output_tensor /= num_microbatches
            forward_data_store.append(loss_reduced)
        else:
            data = loss_func(output_tensor, non_loss_data=True)
            forward_data_store.append(data)

    if config.timers is not None:
        config.timers('forward-compute').stop()

    if run_timer:
        ScheduleTimers.for_chunk(0).f.stop()
        ScheduleTimers.for_chunk(0).f_mem += torch.cuda.memory_allocated() - mem_before

    # Set the loss scale for the auxiliary loss of the MoE layer.
    # Since we use a trick to do backward on the auxiliary loss, we need to set the scale explicitly.
    if hasattr(config, 'num_moe_experts') and config.num_moe_experts is not None:
        # Calculate the loss scale based on the grad_scale_func if available, else default to 1.
        loss_scale = (
            config.grad_scale_func(torch.ones(1, device=output_tensor.device))
            if config.grad_scale_func is not None
            else torch.tensor(1.0)
        )
        # Set the loss scale
        MoEAuxLossAutoScaler.set_loss_scale(loss_scale / num_microbatches)

    # If T5 model and in decoder stack, then send encoder_hidden_state
    # downstream as well.
    model_type = get_model_type(model)
    if (
        model_type == ModelType.encoder_and_decoder
        and encoder_decoder_xattn
        and parallel_state.is_inside_decoder()
    ):
        return [output_tensor, input_tensor[-1]], num_tokens

    if unwrap_output_tensor:
        return output_tensor, num_tokens
    return [output_tensor], num_tokens


profiler_hacker = None


def backward_step(input_tensor, output_tensor, output_tensor_grad, model_type, config,
                  run_timer=False):
    """Backward step through passed-in output tensor.

    If last stage, output_tensor_grad is None, otherwise gradient of loss
    with respect to stage's output tensor.

    Returns gradient of loss with respect to input tensor (None if first
    stage)."""

    # NOTE: This code currently can handle at most one skip connection. It
    # needs to be modified slightly to support arbitrary numbers of skip
    # connections.
    # For mysterious reasons ops generated in autograd doesn't conform to the cuda.nvtx.range.
    # To overcome this we insert a tiny computation in the head & tail of the range

    global profiler_hacker
    if get_args().profile:
        if profiler_hacker is None:
            profiler_hacker = torch.Tensor([0]).cuda()
        profiler_hacker = torch.abs(profiler_hacker)

    # A weird hack, see https://github.com/pytorch/pytorch/issues/124565
    torch.empty(1, device='cuda', requires_grad=True).backward()
    if config.timers is not None:
        config.timers('backward-compute', log_level=2).start()

    if run_timer:
        ScheduleTimers.for_chunk(0).b_cnt += 1
        ScheduleTimers.for_chunk(0).b.start()
    mem_before = torch.cuda.memory_allocated()

    if get_args().enable_exactly_numeric_match:
        reset_random_state()

    # Retain the grad on the input_tensor.
    unwrap_input_tensor_grad = False
    if not isinstance(input_tensor, list):
        input_tensor = [input_tensor]
        unwrap_input_tensor_grad = True
    for x in input_tensor:
        if x is not None:
            x.retain_grad()

    if not isinstance(output_tensor, list):
        output_tensor = [output_tensor]
    if not isinstance(output_tensor_grad, list):
        output_tensor_grad = [output_tensor_grad]

    # Backward pass.
    if output_tensor_grad[0] is None and config.grad_scale_func is not None:
        output_tensor[0] = config.grad_scale_func(output_tensor[0])

    if config.deallocate_pipeline_outputs:
        custom_backward(output_tensor[0], output_tensor_grad[0])
    else:
        torch.autograd.backward(output_tensor[0], grad_tensors=output_tensor_grad[0])

    # Collect the grad of the input_tensor.
    input_tensor_grad = [None]
    if input_tensor is not None:
        input_tensor_grad = []
        for x in input_tensor:
            if x is None:
                input_tensor_grad.append(None)
            else:
                input_tensor_grad.append(x.grad)

    # Handle single skip connection if it exists (encoder_hidden_state in
    # model with encoder and decoder).
    if (
        parallel_state.get_pipeline_model_parallel_world_size() > 1
        and model_type == ModelType.encoder_and_decoder
        and len(output_tensor_grad) > 1  # excludes models that lack a skip connection.
    ):
        if output_tensor_grad[1] is not None:
            assert input_tensor_grad[-1] is not None
            input_tensor_grad[-1].add_(output_tensor_grad[1])
    if unwrap_input_tensor_grad:
        input_tensor_grad = input_tensor_grad[0]

    if run_timer:
        ScheduleTimers.for_chunk(0).b.stop()
        ScheduleTimers.for_chunk(0).b_mem += torch.cuda.memory_allocated() - mem_before

    if config.timers is not None:
        config.timers('backward-compute').stop()
    if get_args().profile:
        profiler_hacker = torch.abs(profiler_hacker)
    return input_tensor_grad


def check_first_val_step(first_val_step, forward_only, cond):
    if (first_val_step is not None) and forward_only:
        return first_val_step and cond
    else:
        return cond


def forward_backward_no_pipelining(
    *,
    forward_step_func,
    data_iterator: Union[Iterator, List[Iterator]],
    model: Union[torch.nn.Module, List[torch.nn.Module]],
    num_microbatches: int,
    seq_length: int,  # unused
    micro_batch_size: int,  # unused
    decoder_seq_length: int = None,  # unused
    forward_only: bool = False,
    collect_non_loss_data: bool = False,
    first_val_step: bool = None,
):
    """Run forward and backward passes with no pipeline parallelism
    (no inter-stage communication).

    Returns dictionary with losses.


    See get_forward_backward_func() for argument details
    """

    if isinstance(model, list):
        assert len(model) == 1, "non-pipeline-parallel schedule does not support model chunking"
        model = model[0]
    if isinstance(data_iterator, list):
        assert (
            len(data_iterator) == 1
        ), "non-pipeline-parallel schedule does not support model chunking"
        data_iterator = data_iterator[0]

    config = get_model_config(model)
    if config.timers is not None:
        config.timers('forward-backward', log_level=1).start(barrier=config.barrier_with_L1_time)

    no_sync_func = config.no_sync_func
    if no_sync_func is None:
        no_sync_func = contextlib.nullcontext

    model_type = get_model_type(model)

    forward_data_store = []
    input_tensor, output_tensor_grad = None, None
    total_num_tokens = torch.zeros([], dtype=torch.int, device="cuda")
    with no_sync_func():
        for i in range(num_microbatches - 1):
            output_tensor, num_tokens = forward_step(
                forward_step_func,
                data_iterator,
                model,
                num_microbatches,
                input_tensor,
                forward_data_store,
                config,
                collect_non_loss_data,
                is_first_microbatch=check_first_val_step(first_val_step, forward_only, i == 0),
                current_microbatch=i,
            )
            total_num_tokens += num_tokens.item()
            if not forward_only:
                backward_step(input_tensor, output_tensor, output_tensor_grad, model_type, config)

    # Run computation for last microbatch out of context handler (want to
    # synchronize gradients).
    output_tensor, num_tokens = forward_step(
        forward_step_func,
        data_iterator,
        model,
        num_microbatches,
        input_tensor,
        forward_data_store,
        config,
        collect_non_loss_data,
        is_first_microbatch=check_first_val_step(
            first_val_step, forward_only, num_microbatches == 1
        ),
        current_microbatch=num_microbatches - 1,
    )
    total_num_tokens += num_tokens.item()

    if not forward_only:
        backward_step(input_tensor, output_tensor, output_tensor_grad, model_type, config)

    if config.finalize_model_grads_func is not None and not forward_only:
        # Finalize model grads (perform full grad all-reduce / reduce-scatter for
        # data parallelism and layernorm all-reduce for sequence parallelism).
        config.finalize_model_grads_func(
            [model], total_num_tokens if config.calculate_per_token_loss else None
        )

    if config.timers is not None:
        config.timers('forward-backward').stop()

    return forward_data_store


def clear_embedding_activation_buffer(config, model):

    if (
        parallel_state.is_pipeline_last_stage(ignore_virtual=True)
        and config.defer_embedding_wgrad_compute
    ):
        if isinstance(model, list):
            embedding_module = get_attr_wrapped_model(
                model[-1], 'post_process', return_model_obj=True
            )
        else:
            embedding_module = get_attr_wrapped_model(model, 'post_process', return_model_obj=True)

        # Need to ensure no stray activations exists in this buffer
        embedding_module.embedding_activation_buffer.clear()

        return embedding_module
    else:
        return None


def finish_embedding_wgrad_compute(config, embedding_module):
    if (
        parallel_state.is_pipeline_last_stage(ignore_virtual=True)
        and config.defer_embedding_wgrad_compute
    ):
        embedding_activation_buffer = embedding_module.embedding_activation_buffer
        grad_output_buffer = embedding_module.grad_output_buffer
        weight = (
            embedding_module.output_layer.weight
            if embedding_module.share_embeddings_and_output_weights
            else embedding_module.shared_embedding_or_output_weight()
        )

        drain_embedding_wgrad_compute(
            config, embedding_activation_buffer, grad_output_buffer, weight
        )


iteration_epoch = 0


def forward_backward_pipelining_with_interleaving(
    *,
    forward_step_func,
    data_iterator: Union[Iterator, List[Iterator]],
    model: Union[torch.nn.Module, List[torch.nn.Module]],
    num_microbatches: int,
    seq_length: int,
    micro_batch_size: int,
    decoder_seq_length: int = None,
    forward_only: bool = False,
    collect_non_loss_data: bool = False,
    first_val_step: bool = None,
):
    """Run interleaved 1F1B schedule (model split into model chunks), with
    communication between pipeline stages as needed.

    Returns dictionary with losses if the last stage, empty dict otherwise."""
    assert isinstance(model, list), "interleaved pipeline parallelism expected model chunking"
    assert all(isinstance(chunk, torch.nn.Module) for chunk in model), "invalid model chunking"
    assert isinstance(
        data_iterator, list
    ), "interleaved pipeline parallelism expected each model chunk to have a data iterator"

    rank = parallel_state.get_pipeline_model_parallel_rank()
    global iteration_epoch
    iteration_epoch += 1
    if get_args().profile_memory_iter >= 0:
        max_allocated = torch.cuda.max_memory_allocated() // 1000000
        current_allocated = torch.cuda.memory_allocated() // 1000000
        print(
            f"MEMORY: {rank} iteration {iteration_epoch} max_allocated: {max_allocated} current_allocated: {current_allocated}")
    if iteration_epoch == get_args().profile_memory_iter:
        torch.cuda.memory._record_memory_history()

    config = get_model_config(model[0])
    if config.overlap_p2p_comm and config.batch_p2p_comm:
        raise ValueError("Can not use both overlap_p2p_comm and batch_p2p_comm")

    # Needed only when gradients are finalized in M-Core
    if config.finalize_model_grads_func is not None and not forward_only:
        embedding_module = clear_embedding_activation_buffer(config, model)

    if config.timers is not None:
        config.timers('forward-backward', log_level=1).start(barrier=config.barrier_with_L1_time)

    # Disable async grad reductions
    no_sync_func = config.no_sync_func
    if isinstance(no_sync_func, list):

        def multi_no_sync():
            stack = contextlib.ExitStack()
            for model_chunk_no_sync_func in config.no_sync_func:
                stack.enter_context(model_chunk_no_sync_func())
            return stack

        no_sync_func = multi_no_sync
    if no_sync_func is None:
        no_sync_func = contextlib.nullcontext
    no_sync_context = None

    if config.grad_sync_func is not None and not isinstance(config.grad_sync_func, list):
        config.grad_sync_func = [config.grad_sync_func for _ in model]

    if config.param_sync_func is not None and not isinstance(config.param_sync_func, list):
        config.param_sync_func = [config.param_sync_func for _ in model]

    def disable_grad_sync():
        """Disable asynchronous grad reductions"""
        nonlocal no_sync_context
        if no_sync_context is None:
            no_sync_context = no_sync_func()
            no_sync_context.__enter__()

    def enable_grad_sync():
        """Enable asynchronous grad reductions"""
        nonlocal no_sync_context
        if no_sync_context is not None:
            no_sync_context.__exit__(None, None, None)
            no_sync_context = None

    disable_grad_sync()

    # Model chunk IDs with synchronized grads
    synchronized_model_chunks = set()

    input_tensors = [[] for _ in range(len(model))]
    output_tensors = [[] for _ in range(len(model))]
    total_num_tokens = torch.tensor(0, dtype=torch.int).cuda()

    forward_data_store = []
    if not forward_only:
        output_tensor_grads = [[] for _ in range(len(model))]

    pipeline_parallel_size = parallel_state.get_pipeline_model_parallel_world_size()
    pipeline_parallel_rank = parallel_state.get_pipeline_model_parallel_rank()

    if num_microbatches % pipeline_parallel_size != 0:
        msg = f'number of microbatches ({num_microbatches}) is not divisible by '
        msg += f'pipeline-model-parallel-size ({pipeline_parallel_size}) '
        msg += 'when using interleaved schedule'
        raise RuntimeError(msg)

    model_type = get_model_type(model[0])
    if model_type == ModelType.encoder_and_decoder:
        raise RuntimeError("Interleaving is not supported with an encoder and decoder model.")

    if decoder_seq_length is not None and decoder_seq_length != seq_length:
        raise RuntimeError(
            "Interleaving is not supported with a different decoder sequence length."
        )

    tensor_shape = [seq_length, micro_batch_size, config.hidden_size]
    tensor_shape[0] = tensor_shape[0] // parallel_state.get_context_parallel_world_size()
    if config.sequence_parallel:
        tensor_shape[0] = tensor_shape[0] // parallel_state.get_tensor_model_parallel_world_size()

    # Compute number of warmup and remaining microbatches.
    num_model_chunks = len(model)
    total_num_microbatches = num_microbatches * num_model_chunks
    all_warmup_microbatches = False
    if forward_only:
        num_warmup_microbatches = total_num_microbatches
    else:
        # Run all forward passes and then all backward passes if number of
        # microbatches is just the number of pipeline stages.
        # Otherwise, perform (num_model_chunks-1)*pipeline_parallel_size on
        # all workers, followed by more microbatches after depending on
        # stage ID (more forward passes for earlier stages, later stages can
        # immediately start with 1F1B).
        if num_microbatches == pipeline_parallel_size:
            num_warmup_microbatches = total_num_microbatches
            all_warmup_microbatches = True
        else:
            num_warmup_microbatches = (pipeline_parallel_size - pipeline_parallel_rank - 1) * 2
            num_warmup_microbatches += (num_model_chunks - 1) * pipeline_parallel_size
            num_warmup_microbatches = min(num_warmup_microbatches, total_num_microbatches)
    num_microbatches_remaining = total_num_microbatches - num_warmup_microbatches

    # Checkpoint the activations of partial Transformer layers in a number of micro-batches
    # within the maximum outstanding micro-batch backpropagations.
    # Micro-batches with the ids less than 'num_microbatches_with_partial_activation_checkpoints'
    # checkpoint partial Transformer layers (or skip checkpointing) and
    # the rest of micro-batches within a window of micro-batches checkpoint
    # all Transformer layers. The window of micro-batches is set by the maximum
    # outstanding backpropagations and becomes smaller at later pipeline stages.
    # Please refer the appendix C in https://arxiv.org/pdf/2205.05198.pdf
    max_outstanding_backprops = None
    if config.num_microbatches_with_partial_activation_checkpoints is not None:
        max_outstanding_backprops = num_warmup_microbatches + 1

    # Synchronize params for first two model chunks
    if config.param_sync_func is not None:
        config.param_sync_func[0](model[0].parameters())
        config.param_sync_func[1](model[1].parameters())

    def get_model_chunk_id(microbatch_id, forward):
        """Helper method to get the model chunk ID given the iteration number."""
        microbatch_id_in_group = microbatch_id % (pipeline_parallel_size * num_model_chunks)
        model_chunk_id = microbatch_id_in_group // pipeline_parallel_size
        if not forward:
            model_chunk_id = num_model_chunks - model_chunk_id - 1
        return model_chunk_id

    def get_microbatch_id_in_model_chunk(iteration_id, forward):
        """Helper method to get the microbatch_id within model chunk given the iteration number."""
        assert forward
        iteration_group_id = iteration_id // (pipeline_parallel_size * num_model_chunks)
        microbatch_id_in_model_chunk = (iteration_group_id * pipeline_parallel_size) + (
            iteration_id % pipeline_parallel_size
        )
        return microbatch_id_in_model_chunk

    def is_first_microbatch_for_model_chunk(microbatch_id: int) -> bool:
        """Check if an iteration is the first for a model chunk."""
        microbatch_group_size = pipeline_parallel_size * num_model_chunks
        num_microbatch_groups = total_num_microbatches // microbatch_group_size
        microbatch_group_id = microbatch_id // microbatch_group_size
        microbatch_id_in_group = microbatch_id % microbatch_group_size
        if microbatch_group_id == 0:
            return microbatch_id_in_group % pipeline_parallel_size == 0
        else:
            return False

    def is_last_microbatch_for_model_chunk(microbatch_id: int) -> bool:
        """Check if an iteration is the last for a model chunk."""
        microbatch_group_size = pipeline_parallel_size * num_model_chunks
        num_microbatch_groups = total_num_microbatches // microbatch_group_size
        microbatch_group_id = microbatch_id // microbatch_group_size
        microbatch_id_in_group = microbatch_id % microbatch_group_size
        if microbatch_group_id == num_microbatch_groups - 1:
            return microbatch_id_in_group % pipeline_parallel_size == pipeline_parallel_size - 1
        else:
            return False

    def forward_step_helper(microbatch_id, current_microbatch, checkpoint_activations_microbatch):
        """Helper method to run forward step with model split into chunks
        (run set_virtual_pipeline_model_parallel_rank() before calling
        forward_step())."""
        model_chunk_id = get_model_chunk_id(microbatch_id, forward=True)
        parallel_state.set_virtual_pipeline_model_parallel_rank(model_chunk_id)

        # launch param synchronization for next model chunk
        # Note: Asynchronous communication tends to slow down compute.
        # To reduce idling from mismatched microbatch times, we launch
        # asynchronous communication at the same time across the
        # pipeline-parallel group.
        if config.param_sync_func is not None:
            param_sync_microbatch_id = microbatch_id + pipeline_parallel_rank
            if (
                param_sync_microbatch_id < total_num_microbatches
                and is_first_microbatch_for_model_chunk(param_sync_microbatch_id)
            ):
                param_sync_chunk_id = get_model_chunk_id(param_sync_microbatch_id, forward=True) + 1
                if 1 < param_sync_chunk_id < num_model_chunks:
                    config.param_sync_func[param_sync_chunk_id](
                        model[param_sync_chunk_id].parameters()
                    )

        # forward step
        if parallel_state.is_pipeline_first_stage():
            if len(input_tensors[model_chunk_id]) == len(output_tensors[model_chunk_id]):
                input_tensors[model_chunk_id].append(None)
        input_tensor = input_tensors[model_chunk_id][-1]

        if get_args().profile:
            torch.cuda.nvtx.range_push('F0.0.0')
        output_tensor, num_tokens = forward_step(
            forward_step_func,
            data_iterator[model_chunk_id],
            model[model_chunk_id],
            num_microbatches,
            input_tensor,
            forward_data_store,
            config,
            collect_non_loss_data,
            checkpoint_activations_microbatch,
            check_first_val_step(
                first_val_step,
                forward_only,
                is_first_microbatch_for_model_chunk(microbatch_id),
            ),
            current_microbatch=current_microbatch,
        )
        if get_args().profile:
            torch.cuda.nvtx.range_pop()
        output_tensors[model_chunk_id].append(output_tensor)

        nonlocal total_num_tokens
        total_num_tokens += num_tokens.item()

        # if forward-only, no need to save tensors for a backward pass
        if forward_only:
            input_tensors[model_chunk_id].pop()
            output_tensors[model_chunk_id].pop()

        return output_tensor

    def backward_step_helper(microbatch_id):
        """Helper method to run backward step with model split into chunks
        (run set_virtual_pipeline_model_parallel_rank() before calling
        backward_step())."""
        model_chunk_id = get_model_chunk_id(microbatch_id, forward=False)
        parallel_state.set_virtual_pipeline_model_parallel_rank(model_chunk_id)

        # launch grad synchronization (default)
        if config.grad_sync_func is None and is_last_microbatch_for_model_chunk(microbatch_id):
            enable_grad_sync()
            synchronized_model_chunks.add(model_chunk_id)

        if parallel_state.is_pipeline_last_stage():
            if len(output_tensor_grads[model_chunk_id]) == 0:
                output_tensor_grads[model_chunk_id].append(None)
        input_tensor = input_tensors[model_chunk_id].pop(0)
        output_tensor = output_tensors[model_chunk_id].pop(0)
        output_tensor_grad = output_tensor_grads[model_chunk_id].pop(0)
        if get_args().profile:
            torch.cuda.nvtx.range_push('B0.0.0')
        input_tensor_grad = backward_step(
            input_tensor, output_tensor, output_tensor_grad, model_type, config
        )
        if get_args().profile:
            torch.cuda.nvtx.range_pop()

        # launch grad synchronization (custom grad sync)
        # Note: Asynchronous communication tends to slow down compute.
        # To reduce idling from mismatched microbatch times, we launch
        # asynchronous communication at the same time across the
        # pipeline-parallel group.
        if config.grad_sync_func is not None:
            grad_sync_microbatch_id = microbatch_id - pipeline_parallel_rank
            if grad_sync_microbatch_id >= 0 and is_last_microbatch_for_model_chunk(
                grad_sync_microbatch_id
            ):
                grad_sync_chunk_id = get_model_chunk_id(grad_sync_microbatch_id, forward=False)
                enable_grad_sync()
                config.grad_sync_func[grad_sync_chunk_id](model[grad_sync_chunk_id].parameters())
                synchronized_model_chunks.add(grad_sync_chunk_id)
        disable_grad_sync()

        return input_tensor_grad

    # Run warmup forward passes.
    parallel_state.set_virtual_pipeline_model_parallel_rank(0)
    input_tensors[0].append(p2p_communication.recv_forward(tensor_shape, config))

    fwd_wait_handles = None
    bwd_wait_handles = None

    for k in range(num_warmup_microbatches):

        if fwd_wait_handles is not None:
            for req in fwd_wait_handles:
                req.wait()

        cur_model_chunk_id = get_model_chunk_id(k, forward=True)
        # Decide to checkpoint all layers' activations of the current micro-batch
        if max_outstanding_backprops is not None:
            checkpoint_activations_microbatch = (
                k % max_outstanding_backprops
                >= config.num_microbatches_with_partial_activation_checkpoints
            )
        else:
            checkpoint_activations_microbatch = None

        current_microbatch = get_microbatch_id_in_model_chunk(k, forward=True)
        output_tensor = forward_step_helper(
            k, current_microbatch, checkpoint_activations_microbatch
        )

        # Determine if tensor should be received from previous stage.
        next_forward_model_chunk_id = get_model_chunk_id(k + 1, forward=True)
        recv_prev = True
        if parallel_state.is_pipeline_first_stage(ignore_virtual=True):
            if next_forward_model_chunk_id == 0:
                recv_prev = False
        if k == (total_num_microbatches - 1):
            recv_prev = False

        # Don't send tensor downstream if on last stage.
        if parallel_state.is_pipeline_last_stage():
            output_tensor = None

        # Send and receive tensors as appropriate (send tensors computed
        # in this iteration; receive tensors for next iteration).
        if not config.overlap_p2p_comm:
            if (
                k == (num_warmup_microbatches - 1)
                and not forward_only
                and not all_warmup_microbatches
            ):
                input_tensor_grad = None
                recv_next = True
                if parallel_state.is_pipeline_last_stage(ignore_virtual=True):
                    recv_next = False
                (
                    input_tensor,
                    output_tensor_grad,
                ) = p2p_communication.send_forward_backward_recv_forward_backward(
                    output_tensor,
                    input_tensor_grad,
                    recv_prev=recv_prev,
                    recv_next=recv_next,
                    tensor_shape=tensor_shape,
                    config=config,
                )
                output_tensor_grads[num_model_chunks - 1].append(output_tensor_grad)
            else:
                input_tensor = p2p_communication.send_forward_recv_forward(
                    output_tensor, recv_prev=recv_prev, tensor_shape=tensor_shape, config=config
                )
            input_tensors[next_forward_model_chunk_id].append(input_tensor)
        else:
            input_tensor, fwd_wait_handles = p2p_communication.send_forward_recv_forward(
                output_tensor,
                recv_prev=recv_prev,
                tensor_shape=tensor_shape,
                config=config,
                overlap_p2p_comm=True,
            )

            if (
                k == (num_warmup_microbatches - 1)
                and not forward_only
                and not all_warmup_microbatches
            ):
                input_tensor_grad = None
                recv_next = True
                if parallel_state.is_pipeline_last_stage(ignore_virtual=True):
                    recv_next = False

                (
                    output_tensor_grad,
                    bwd_wait_handles,
                ) = p2p_communication.send_backward_recv_backward(
                    input_tensor_grad,
                    recv_next=recv_next,
                    tensor_shape=tensor_shape,
                    config=config,
                    overlap_p2p_comm=True,
                )

                output_tensor_grads[num_model_chunks - 1].append(output_tensor_grad)
            input_tensors[next_forward_model_chunk_id].append(input_tensor)

        deallocate_output_tensor(output_tensor, config.deallocate_pipeline_outputs)

    # Run 1F1B in steady state.
    for k in range(num_microbatches_remaining):
        # Forward pass.
        forward_k = k + num_warmup_microbatches

        # Decide to checkpoint all layers' activations of the current micro-batch
        if max_outstanding_backprops is not None:
            checkpoint_activations_microbatch = (
                forward_k % max_outstanding_backprops
                >= config.num_microbatches_with_partial_activation_checkpoints
            )
        else:
            checkpoint_activations_microbatch = None

        cur_model_chunk_id = get_model_chunk_id(forward_k, forward=True)
        current_microbatch = get_microbatch_id_in_model_chunk(forward_k, forward=True)
        if config.overlap_p2p_comm:
            if fwd_wait_handles is not None:
                for req in fwd_wait_handles:
                    req.wait()

            deallocate_output_tensor(output_tensor, config.deallocate_pipeline_outputs)

            output_tensor = forward_step_helper(
                forward_k, current_microbatch, checkpoint_activations_microbatch
            )

            # Determine if current stage has anything to send in either direction,
            # otherwise set tensor to None.
            forward_model_chunk_id = get_model_chunk_id(forward_k, forward=True)
            parallel_state.set_virtual_pipeline_model_parallel_rank(forward_model_chunk_id)

            # Last virtual stage no activation tensor to send
            if parallel_state.is_pipeline_last_stage():
                output_tensor = None

            # Determine if peers are sending, and where in data structure to put
            # received tensors.
            recv_prev = True
            if parallel_state.is_pipeline_first_stage(ignore_virtual=True):
                # First stage is ahead of last stage by (pipeline_parallel_size - 1).
                next_forward_model_chunk_id = get_model_chunk_id(
                    forward_k - (pipeline_parallel_size - 1), forward=True
                )
                if next_forward_model_chunk_id == (num_model_chunks - 1):
                    recv_prev = False
                next_forward_model_chunk_id += 1
            else:
                next_forward_model_chunk_id = get_model_chunk_id(forward_k + 1, forward=True)

            # If last iteration, don't receive; we already received one extra
            # before the start of the for loop.
            if k == (num_microbatches_remaining - 1):
                recv_prev = False

            # Send activation tensor to the next stage and receive activation tensor from the
            # previous stage
            input_tensor, fwd_wait_handles = p2p_communication.send_forward_recv_forward(
                output_tensor,
                recv_prev=recv_prev,
                tensor_shape=tensor_shape,
                config=config,
                overlap_p2p_comm=True,
            )
            # assert fwd_wait_handles is not None

            if bwd_wait_handles is not None:
                for req in bwd_wait_handles:
                    req.wait()

            # Backward pass.
            backward_k = k
            input_tensor_grad = backward_step_helper(backward_k)

            backward_model_chunk_id = get_model_chunk_id(backward_k, forward=False)
            parallel_state.set_virtual_pipeline_model_parallel_rank(backward_model_chunk_id)

            # First virtual stage no activation gradient tensor to send
            if parallel_state.is_pipeline_first_stage():
                input_tensor_grad = None

            # Determine if the current virtual stage has an activation gradient tensor to receive
            recv_next = True
            if parallel_state.is_pipeline_last_stage(ignore_virtual=True):
                # Last stage is ahead of first stage by (pipeline_parallel_size - 1).
                next_backward_model_chunk_id = get_model_chunk_id(
                    backward_k - (pipeline_parallel_size - 1), forward=False
                )
                if next_backward_model_chunk_id == 0:
                    recv_next = False
                next_backward_model_chunk_id -= 1
            else:
                next_backward_model_chunk_id = get_model_chunk_id(backward_k + 1, forward=False)

            output_tensor_grad, bwd_wait_handles = p2p_communication.send_backward_recv_backward(
                input_tensor_grad,
                recv_next=recv_next,
                tensor_shape=tensor_shape,
                config=config,
                overlap_p2p_comm=True,
            )

        else:  # no p2p overlap
            output_tensor = forward_step_helper(
                forward_k, current_microbatch, checkpoint_activations_microbatch
            )

            # Backward pass.
            backward_k = k
            input_tensor_grad = backward_step_helper(backward_k)

            # Send output_tensor and input_tensor_grad, receive input_tensor
            # and output_tensor_grad.

            # Determine if current stage has anything to send in either direction,
            # otherwise set tensor to None.
            forward_model_chunk_id = get_model_chunk_id(forward_k, forward=True)
            parallel_state.set_virtual_pipeline_model_parallel_rank(forward_model_chunk_id)
            if parallel_state.is_pipeline_last_stage():
                output_tensor = None

            backward_model_chunk_id = get_model_chunk_id(backward_k, forward=False)
            parallel_state.set_virtual_pipeline_model_parallel_rank(backward_model_chunk_id)
            if parallel_state.is_pipeline_first_stage():
                input_tensor_grad = None

            # Determine if peers are sending, and where in data structure to put
            # received tensors.
            recv_prev = True
            if parallel_state.is_pipeline_first_stage(ignore_virtual=True):
                # First stage is ahead of last stage by (pipeline_parallel_size - 1).
                next_forward_model_chunk_id = get_model_chunk_id(
                    forward_k - (pipeline_parallel_size - 1), forward=True
                )
                if next_forward_model_chunk_id == (num_model_chunks - 1):
                    recv_prev = False
                next_forward_model_chunk_id += 1
            else:
                next_forward_model_chunk_id = get_model_chunk_id(forward_k + 1, forward=True)

            recv_next = True
            if parallel_state.is_pipeline_last_stage(ignore_virtual=True):
                # Last stage is ahead of first stage by (pipeline_parallel_size - 1).
                next_backward_model_chunk_id = get_model_chunk_id(
                    backward_k - (pipeline_parallel_size - 1), forward=False
                )
                if next_backward_model_chunk_id == 0:
                    recv_next = False
                next_backward_model_chunk_id -= 1
            else:
                next_backward_model_chunk_id = get_model_chunk_id(backward_k + 1, forward=False)

            # If last iteration, don't receive; we already received one extra
            # before the start of the for loop.
            if k == (num_microbatches_remaining - 1):
                recv_prev = False

            # Communicate tensors.
            (
                input_tensor,
                output_tensor_grad,
            ) = p2p_communication.send_forward_backward_recv_forward_backward(
                output_tensor,
                input_tensor_grad,
                recv_prev=recv_prev,
                recv_next=recv_next,
                tensor_shape=tensor_shape,
                config=config,
            )
            deallocate_output_tensor(output_tensor, config.deallocate_pipeline_outputs)

        # Put input_tensor and output_tensor_grad in data structures in the
        # right location.
        if recv_prev:
            input_tensors[next_forward_model_chunk_id].append(input_tensor)
        if recv_next:
            output_tensor_grads[next_backward_model_chunk_id].append(output_tensor_grad)

    deallocate_output_tensor(output_tensor, config.deallocate_pipeline_outputs)

    # Run cooldown backward passes (flush out pipeline).
    if not forward_only:
        if config.overlap_p2p_comm and bwd_wait_handles is not None:
            for wait_handle in bwd_wait_handles:
                wait_handle.wait()

        if all_warmup_microbatches:
            output_tensor_grads[num_model_chunks - 1].append(
                p2p_communication.recv_backward(tensor_shape, config=config)
            )
        for k in range(num_microbatches_remaining, total_num_microbatches):
            input_tensor_grad = backward_step_helper(k)
            next_backward_model_chunk_id = get_model_chunk_id(k + 1, forward=False)
            recv_next = True
            if parallel_state.is_pipeline_last_stage(ignore_virtual=True):
                if next_backward_model_chunk_id == (num_model_chunks - 1):
                    recv_next = False
            if k == (total_num_microbatches - 1):
                recv_next = False
            output_tensor_grads[next_backward_model_chunk_id].append(
                p2p_communication.send_backward_recv_backward(
                    input_tensor_grad, recv_next=recv_next, tensor_shape=tensor_shape, config=config
                )
            )

        # Launch any remaining grad reductions.
        enable_grad_sync()
        if config.grad_sync_func is not None:
            for model_chunk_id in range(num_model_chunks):
                if model_chunk_id not in synchronized_model_chunks:
                    config.grad_sync_func[model_chunk_id](model[model_chunk_id].parameters())
                    synchronized_model_chunks.add(model_chunk_id)

    if config.finalize_model_grads_func is not None and not forward_only:

        # If defer_embedding_wgrad_compute is enabled we need to do the
        # weight gradient GEMM's here.
        finish_embedding_wgrad_compute(config, embedding_module)

        # Finalize model grads (perform full grad all-reduce / reduce-scatter for
        # data parallelism, layernorm all-reduce for sequence parallelism, and
        # embedding all-reduce for pipeline parallelism).
        config.finalize_model_grads_func(
            model, total_num_tokens if config.calculate_per_token_loss else None
        )

    if config.timers is not None:
        config.timers('forward-backward').stop()

    if iteration_epoch == get_args().profile_memory_iter:
        torch.cuda.memory._dump_snapshot(f"mem-profile-i1F1B-rank{rank}")
    return forward_data_store


def get_tensor_shapes(
    *,
    rank: int,
    model_type: ModelType,
    seq_length: int,
    micro_batch_size: int,
    decoder_seq_length: int,
    config,
    encoder_decoder_xattn: bool,
):
    # Determine right tensor sizes (based on position of rank with respect to split rank) and model size.
    # Send two tensors if model decoder requires the encoder's output (via cross-attention) and rank is in decoder stage.
    #     first tensor is decoder.
    #     second tensor is encoder.
    # If model has an encoder & decoder and rank is at the boundary:
    #     send one tensor.
    # Otherwise, send one tensor.
    tensor_shapes = []

    seq_length = seq_length // parallel_state.get_context_parallel_world_size()
    if model_type == ModelType.encoder_and_decoder:
        decoder_seq_length = decoder_seq_length // parallel_state.get_context_parallel_world_size()

    if config.sequence_parallel:
        seq_length = seq_length // parallel_state.get_tensor_model_parallel_world_size()
        if model_type == ModelType.encoder_and_decoder:
            decoder_seq_length = (
                decoder_seq_length // parallel_state.get_tensor_model_parallel_world_size()
            )

    if model_type == ModelType.encoder_and_decoder:
        if parallel_state.is_inside_encoder(rank):
            tensor_shapes.append((seq_length, micro_batch_size, config.hidden_size))
        elif encoder_decoder_xattn:
            tensor_shapes.append((decoder_seq_length, micro_batch_size, config.hidden_size))
            tensor_shapes.append((seq_length, micro_batch_size, config.hidden_size))
        else:
            tensor_shapes.append((decoder_seq_length, micro_batch_size, config.hidden_size))
    else:  # model_type == ModelType.encoder_or_decoder
        tensor_shapes.append((seq_length, micro_batch_size, config.hidden_size))
    return tensor_shapes


def recv_forward(tensor_shapes, config):
    input_tensors = []
    for tensor_shape in tensor_shapes:
        if tensor_shape is None:
            input_tensors.append(None)
        else:
            input_tensors.append(p2p_communication.recv_forward(tensor_shape, config))
    return input_tensors


def recv_backward(tensor_shapes, config):
    output_tensor_grads = []
    for tensor_shape in tensor_shapes:
        if tensor_shape is None:
            output_tensor_grads.append(None)
        else:
            output_tensor_grads.append(p2p_communication.recv_backward(tensor_shape, config))
    return output_tensor_grads


def send_forward(output_tensors, tensor_shapes, config):
    if not isinstance(output_tensors, list):
        output_tensors = [output_tensors]
    for output_tensor, tensor_shape in zip(output_tensors, tensor_shapes):
        if tensor_shape is None:
            continue
        p2p_communication.send_forward(output_tensor, config)


def send_backward(input_tensor_grads, tensor_shapes, config):
    if not isinstance(input_tensor_grads, list):
        input_tensor_grads = [input_tensor_grads]
    for input_tensor_grad, tensor_shape in zip(input_tensor_grads, tensor_shapes):
        if tensor_shape is None:
            continue
        p2p_communication.send_backward(input_tensor_grad, config)


def send_forward_recv_backward(output_tensors, tensor_shapes, config):
    if not isinstance(output_tensors, list):
        output_tensors = [output_tensors]
    output_tensor_grads = []
    for output_tensor, tensor_shape in zip(output_tensors, tensor_shapes):
        if tensor_shape is None:
            output_tensor_grads.append(None)
            continue
        output_tensor_grad = p2p_communication.send_forward_recv_backward(
            output_tensor, tensor_shape, config
        )
        output_tensor_grads.append(output_tensor_grad)
    return output_tensor_grads


def send_backward_recv_forward(input_tensor_grads, tensor_shapes, config):
    if not isinstance(input_tensor_grads, list):
        input_tensor_grads = [input_tensor_grads]
    input_tensors = []
    for input_tensor_grad, tensor_shape in zip(input_tensor_grads, tensor_shapes):
        if tensor_shape is None:
            input_tensors.append(None)
            continue
        input_tensor = p2p_communication.send_backward_recv_forward(
            input_tensor_grad, tensor_shape, config
        )
        input_tensors.append(input_tensor)
    return input_tensors


def forward_backward_pipelining_without_interleaving(
    *,
    forward_step_func,
    data_iterator: Union[Iterator, List[Iterator]],
    model: Union[torch.nn.Module, List[torch.nn.Module]],
    num_microbatches: int,
    seq_length: int,
    micro_batch_size: int,
    decoder_seq_length: int = None,
    forward_only: bool = False,
    collect_non_loss_data: bool = False,
    first_val_step: bool = None,
):
    """Run non-interleaved 1F1B schedule, with communication between pipeline
    stages.

    Returns dictionary with losses if the last stage, empty dict otherwise."""

    if isinstance(model, list):
        assert (
            len(model) == 1
        ), "non-interleaved pipeline parallelism does not support model chunking"
        model = model[0]
    if isinstance(data_iterator, list):
        assert (
            len(data_iterator) == 1
        ), "non-pipeline-parallel schedule does not support model chunking"
        data_iterator = data_iterator[0]

    config = get_model_config(model)
    if config.overlap_p2p_comm:
        raise ValueError(
            "Non-interleaved pipeline parallelism does not support overlapping p2p communication"
        )

    # Needed only when gradients are finalized in M-Core
    if config.finalize_model_grads_func is not None and not forward_only:
        embedding_module = clear_embedding_activation_buffer(config, model)

    if config.timers is not None:
        config.timers('forward-backward', log_level=1).start(barrier=config.barrier_with_L1_time)

    # Disable async grad reductions
    no_sync_func = config.no_sync_func
    if no_sync_func is None:
        no_sync_func = contextlib.nullcontext
    no_sync_context = None

    def disable_grad_sync():
        """Disable asynchronous grad reductions"""
        nonlocal no_sync_context
        if no_sync_context is None:
            no_sync_context = no_sync_func()
            no_sync_context.__enter__()

    def enable_grad_sync():
        """Enable asynchronous grad reductions"""
        nonlocal no_sync_context
        if no_sync_context is not None:
            no_sync_context.__exit__(None, None, None)
            no_sync_context = None

    disable_grad_sync()

    # Compute number of warmup microbatches.
    num_warmup_microbatches = (
        parallel_state.get_pipeline_model_parallel_world_size()
        - parallel_state.get_pipeline_model_parallel_rank()
        + get_args().num_seq_splits - 2 # Hack 1f1b to something like seq 1f1b but 1/2 of the microbatches
    )
    num_warmup_microbatches = min(num_warmup_microbatches, num_microbatches)
    num_microbatches_remaining = num_microbatches - num_warmup_microbatches

    # Checkpoint the activations of partial Transformer layers in a number of micro-batches
    # within the maximum outstanding micro-batch backpropagations.
    # Micro-batches with the ids less than 'num_microbatches_with_partial_activation_checkpoints'
    # checkpoint partial Transformer layers (or skip checkpointing) and
    # the rest of micro-batches within a window of micro-batches checkpoint
    # all Transformer layers. The window of micro-batches is set by the maximum
    # outstanding backpropagations and becomes smaller at later pipeline stages.
    # Please refer the appendix C in https://arxiv.org/pdf/2205.05198.pdf
    max_outstanding_backprops = None
    if config.num_microbatches_with_partial_activation_checkpoints is not None:
        max_outstanding_backprops = num_warmup_microbatches + 1

    model_type = get_model_type(model)
    encoder_decoder_xattn = get_model_xattn(model)

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
    send_tensor_shapes = get_tensor_shapes(
        rank=rank,
        model_type=model_type,
        seq_length=seq_length,
        micro_batch_size=micro_batch_size,
        decoder_seq_length=decoder_seq_length,
        config=config,
        encoder_decoder_xattn=encoder_decoder_xattn,
    )

    # Input, output tensors only need to be saved when doing backward passes
    input_tensors = None
    output_tensors = None
    total_num_tokens = torch.tensor(0, dtype=torch.int).cuda()


    # A queue of a stack
    class SpQueue:
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

    if not forward_only:
        input_tensors = SpQueue(get_args().num_seq_splits)
        output_tensors = SpQueue(get_args().num_seq_splits)
    forward_data_store = []

    # Run warmup forward passes.
    for i in range(num_warmup_microbatches):
        # Decide to checkpoint all layers' activations of the current micro-batch
        if max_outstanding_backprops is not None:
            checkpoint_activations_microbatch = (
                i % max_outstanding_backprops
                >= config.num_microbatches_with_partial_activation_checkpoints
            )
        else:
            checkpoint_activations_microbatch = None

        input_tensor = recv_forward(recv_tensor_shapes, config)
        parallel_state.set_seq_split_idx(i % get_args().num_seq_splits)
        output_tensor, num_tokens = forward_step(
            forward_step_func,
            data_iterator,
            model,
            num_microbatches,
            input_tensor,
            forward_data_store,
            config,
            collect_non_loss_data,
            checkpoint_activations_microbatch,
            check_first_val_step(first_val_step, forward_only, i == 0),
            current_microbatch=i,
            encoder_decoder_xattn=encoder_decoder_xattn,
        )
        send_forward(output_tensor, send_tensor_shapes, config)
        total_num_tokens += num_tokens.item()

        if not forward_only:
            input_tensors.push(input_tensor)
            output_tensors.push(output_tensor)
            deallocate_output_tensor(output_tensor[0], config.deallocate_pipeline_outputs)

    # Before running 1F1B, need to receive first forward tensor.
    # If all microbatches are run in warmup / cooldown phase, then no need to
    # receive this tensor here.
    if num_microbatches_remaining > 0:
        input_tensor = recv_forward(recv_tensor_shapes, config)

    from megatron.core.zbpp_utils import WeightGradStore
    WeightGradStore.disable_split_bw()

    # Run 1F1B in steady state.
    for i in range(num_microbatches_remaining):
        last_iteration = i == (num_microbatches_remaining - 1)

        # Decide to checkpoint all layers' activations of the current micro-batch
        if max_outstanding_backprops is not None:
            checkpoint_activations_microbatch = (
                (i + num_warmup_microbatches) % max_outstanding_backprops
            ) >= config.num_microbatches_with_partial_activation_checkpoints
        else:
            checkpoint_activations_microbatch = None
        parallel_state.set_seq_split_idx((i + num_warmup_microbatches) % get_args().num_seq_splits)
        output_tensor, num_tokens = forward_step(
            forward_step_func,
            data_iterator,
            model,
            num_microbatches,
            input_tensor,
            forward_data_store,
            config,
            collect_non_loss_data,
            checkpoint_activations_microbatch,
            check_first_val_step(
                first_val_step, forward_only, (i == 0) and (num_warmup_microbatches == 0)
            ),
            current_microbatch=i + num_warmup_microbatches,
            encoder_decoder_xattn=encoder_decoder_xattn,
        )
        total_num_tokens += num_tokens.item()

        if forward_only:
            send_forward(output_tensor, send_tensor_shapes, config)

            if not last_iteration:
                input_tensor = recv_forward(recv_tensor_shapes, config)

        else:
            output_tensor_grad = send_forward_recv_backward(
                output_tensor, send_tensor_shapes, config
            )

            # Add input_tensor and output_tensor to end of list.
            input_tensors.push(input_tensor)
            output_tensors.push(output_tensor)
            deallocate_output_tensor(output_tensor[0], config.deallocate_pipeline_outputs)

            # Pop input_tensor and output_tensor from the start of the list for
            # the backward pass.
            input_tensor = input_tensors.pop()
            output_tensor = output_tensors.pop()

            # Enable grad sync for the last microbatch in the batch if the full
            # backward pass completes in the 1F1B stage.
            if num_warmup_microbatches == 0 and last_iteration:
                if config.grad_sync_func is None or rank == 0:
                    enable_grad_sync()

            # Selectively enabling bw-split will change the order of W,
            # making it not exactly match the origin numeric results.
            # So disable it when enable_exactly_numeric_match is true.
            if (i < rank or last_iteration) and rank > 0 and not get_args().enable_exactly_numeric_match:
                WeightGradStore.enable_split_bw()
            else:
                WeightGradStore.disable_split_bw()

            input_tensor_grad = backward_step(
                input_tensor, output_tensor, output_tensor_grad, model_type, config
            )

            if WeightGradStore.split_bw():
                WeightGradStore.flush()

            if last_iteration:
                input_tensor = None
                send_backward(input_tensor_grad, recv_tensor_shapes, config)
                if i >= rank > 0 and not get_args().enable_exactly_numeric_match:  # delay W by rank
                    WeightGradStore.pop()  # W
            else:
                input_tensor = send_backward_recv_forward(
                    input_tensor_grad, recv_tensor_shapes, config
                )

    # Run cooldown backward passes.
    if not forward_only:
        for i in range(num_warmup_microbatches):

            # Enable async grad reduction in the last backward pass
            # Note: If grad sync function is provided, only enable
            # async grad reduction in first pipeline stage. Other
            # pipeline stages do grad reduction during pipeline
            # bubble.
            if i == num_warmup_microbatches - 1:
                if config.grad_sync_func is None or rank == 0:
                    enable_grad_sync()

            input_tensor = input_tensors.pop()
            output_tensor = output_tensors.pop()

            output_tensor_grad = recv_backward(send_tensor_shapes, config)

            if rank > 0 and not get_args().enable_exactly_numeric_match:
                WeightGradStore.enable_split_bw()
            else:
                WeightGradStore.disable_split_bw()

            input_tensor_grad = backward_step(
                input_tensor, output_tensor, output_tensor_grad, model_type, config
            )

            send_backward(input_tensor_grad, recv_tensor_shapes, config)

            if WeightGradStore.split_bw():
                WeightGradStore.flush()
                if num_microbatches_remaining + i >= rank:
                    WeightGradStore.pop()  # W

        WeightGradStore.clear(model)

        # Launch any remaining grad reductions.
        if no_sync_context is not None:
            enable_grad_sync()
            if config.grad_sync_func is not None:
                config.grad_sync_func(model.parameters())

    if config.finalize_model_grads_func is not None and not forward_only:

        # If defer_embedding_wgrad_compute is enabled we need to do the
        # weight gradient GEMM's here.
        finish_embedding_wgrad_compute(config, embedding_module)

        # Finalize model grads (perform full grad all-reduce / reduce-scatter for
        # data parallelism, layernorm all-reduce for sequence parallelism, and
        # embedding all-reduce for pipeline parallelism).
        config.finalize_model_grads_func(
            [model], total_num_tokens if config.calculate_per_token_loss else None
        )

    if config.timers is not None:
        config.timers('forward-backward').stop()

    return forward_data_store


def bootstrap_and_profile_p2p_communication(
    config, send_tensor_shapes, recv_tensor_shapes
):
    if ScheduleTimers.iter_counter == 1:
        nccl_init_tensor = [torch.Tensor([0]).cuda()]
        shape = [(1,)]
        if not parallel_state.is_pipeline_first_stage(ignore_virtual=True):
            recv_forward(shape, config)
        if not parallel_state.is_pipeline_last_stage(ignore_virtual=True):
            send_forward(nccl_init_tensor, shape, config)
            recv_backward(shape, config)
        if not parallel_state.is_pipeline_first_stage(ignore_virtual=True):
            send_backward(nccl_init_tensor, shape, config)

        send_data = [torch.zeros(*shape, dtype=config.pipeline_dtype).cuda() for
                     shape in send_tensor_shapes]
        recv_data = [torch.zeros(*shape, dtype=config.pipeline_dtype).cuda() for
                     shape in recv_tensor_shapes]
        torch.distributed.barrier()
        t = Timer('comm-benchmark')
        t.start()
        for _ in range(10):
            if not parallel_state.is_pipeline_first_stage(ignore_virtual=True):
                recv_forward(recv_tensor_shapes, config)
            if not parallel_state.is_pipeline_last_stage(ignore_virtual=True):
                send_forward(send_data, send_tensor_shapes, config)
                recv_backward(send_tensor_shapes, config)
            if not parallel_state.is_pipeline_first_stage(ignore_virtual=True):
                send_backward(recv_data, recv_tensor_shapes, config)
        t.stop()
        per_communication = torch.cuda.FloatTensor([t.elapsed() / (
            parallel_state.get_pipeline_model_parallel_world_size() - 1) / 10])
        torch.distributed.all_reduce(per_communication, torch.distributed.ReduceOp.MAX)
        ScheduleTimers.comm_time = per_communication.item()

        global LM_HEAD_RES_REDUCE_STREAM
        LM_HEAD_RES_REDUCE_STREAM = torch.cuda.Stream()



def forward_backward_pipelining_with_split_vocab_parallel(
    *,
    forward_step_func,
    data_iterator: Union[Iterator, List[Iterator]],
    model: Union[torch.nn.Module, List[torch.nn.Module]],
    num_microbatches: int,
    seq_length: int,
    micro_batch_size: int,
    decoder_seq_length: int = None,
    forward_only: bool = False,
    collect_non_loss_data: bool = False,
    first_val_step: bool = None,
):
    """Run non-interleaved 1F1B schedule with language model head layer sharding
    on the vocabulary dimension, with communication between pipeline stages.

    Returns dictionary with losses if the last stage, empty dict otherwise."""

    assert isinstance(model, list)

    config = get_model_config(model[0])
    if config.overlap_p2p_comm:
        raise ValueError(
            "Non-interleaved pipeline parallelism does not support overlapping p2p communication"
        )
    
    assert not forward_only, "Vocab parallel is incompatible with forward only."

    # Needed only when gradients are finalized in M-Core
    if config.finalize_model_grads_func is not None and not forward_only:
        embedding_module = clear_embedding_activation_buffer(config, model[0])

    if config.timers is not None:
        config.timers('forward-backward', log_level=1).start(barrier=config.barrier_with_L1_time)

    # Disable async grad reductions
    no_sync_func = config.no_sync_func
    if no_sync_func is None:
        no_sync_func = contextlib.nullcontext
    no_sync_context = None

    def disable_grad_sync():
        """Disable asynchronous grad reductions"""
        nonlocal no_sync_context
        if no_sync_context is None:
            no_sync_context = no_sync_func()
            no_sync_context.__enter__()

    def enable_grad_sync():
        """Enable asynchronous grad reductions"""
        nonlocal no_sync_context
        if no_sync_context is not None:
            no_sync_context.__exit__(None, None, None)
            no_sync_context = None

    disable_grad_sync()

    # Increment iter_counter in ScheduleTimers
    ScheduleTimers.iter_counter += 1

    if ScheduleTimers.iter_counter == get_args().schedule_timer_end + 1:
        ScheduleTimers.sync_timer = False

    if ScheduleTimers.iter_counter == get_args().schedule_timer_end + 6:
        conclusion = ScheduleTimers.joint_conclusion(sync_timer=False, global_reduce=False)
        print(f"rank {torch.distributed.get_rank()} profiling conclusion: {conclusion}")
    
    if ScheduleTimers.iter_counter >= get_args().schedule_timer_end + 1:
        conclusion = ScheduleTimers.joint_conclusion()
        f = conclusion[0][0][0]
        b = conclusion[0][0][1]
        c = conclusion[0][0][3]
    else:
        f = 1
        b = 2
        c = 0
    
    assert f >= c, 'vocab parallel schedules assume f >= c, ' \
        'f < c will lead to additional pipeline bubbles due to incorrect ' \
        'placements for the S pass'
    assert b >= f, 'vocab parallel schedules assume b >= f for S pass placements'
    num_stages = parallel_state.get_pipeline_model_parallel_world_size()
    offset = 0
    is_bsf = [True]
    offsets = [0]
    while len(is_bsf) < num_stages:
        # we can either subtract f from the offset, or add (b - f) to the offset
        # FSM:
        # - BSF --[ - f ]--> BSF
        # - BSF --[ 0 ]--> BFS --[ (b - f) ]--> BSF
        if offset - f >= -b:
            offset = offset - f
            is_bsf.append(True)
            offsets.append(offset)
        else:
            is_bsf.append(False)
            offsets.append(offset)
            offset = offset + (b - f)
            is_bsf.append(True)
            offsets.append(offset)
    if len(is_bsf) > num_stages:
        is_bsf.pop()
        offsets.pop()
    
    is_bsf.reverse()
    offsets.reverse()

    num_warmup_s_pass = [0 for _ in range(num_stages)]
    for rank in range(num_stages - 2, -1, -1):
        if (not is_bsf[rank + 1]) and (is_bsf[rank]):
            num_warmup_s_pass[rank] = num_warmup_s_pass[rank + 1]
        else:
            num_warmup_s_pass[rank] = num_warmup_s_pass[rank + 1] + 1

    run_timer = (
        get_args().schedule_timer_end + 5
        >= ScheduleTimers.iter_counter
        >= get_args().schedule_timer_start
    )

    # Compute number of warmup microbatches.
    num_warmup_microbatches = (
        parallel_state.get_pipeline_model_parallel_world_size()
        - parallel_state.get_pipeline_model_parallel_rank()
    )
    num_warmup_microbatches = min(num_warmup_microbatches, num_microbatches)
    first_stage_num_warmup_microbatches = min(
        parallel_state.get_pipeline_model_parallel_world_size(),
        num_microbatches,
    )
    num_microbatches_remaining = max(
        0,
        num_microbatches - num_warmup_microbatches - 2
    )

    assert config.num_microbatches_with_partial_activation_checkpoints is None, 'not supported'

    model_type = get_model_type(model[0])
    encoder_decoder_xattn = get_model_xattn(model[0])

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
    send_tensor_shapes = get_tensor_shapes(
        rank=rank,
        model_type=model_type,
        seq_length=seq_length,
        micro_batch_size=micro_batch_size,
        decoder_seq_length=decoder_seq_length,
        config=config,
        encoder_decoder_xattn=encoder_decoder_xattn,
    )
    lm_head_tensor_shapes = get_tensor_shapes(
        rank=parallel_state.get_pipeline_model_parallel_world_size() - 1,
        model_type=model_type,
        seq_length=seq_length,
        micro_batch_size=micro_batch_size,
        decoder_seq_length=decoder_seq_length,
        config=config,
        encoder_decoder_xattn=encoder_decoder_xattn,
    )

    bootstrap_and_profile_p2p_communication(
        config, send_tensor_shapes, recv_tensor_shapes)

    # Input, output tensors only need to be saved when doing backward passes
    input_tensors = None
    output_tensors = None
    total_num_tokens = torch.tensor(0, dtype=torch.int).cuda()

    input_tensors = [[], [], []]
    output_tensors = [[], [], []]
    forward_data_store = []

    # Storing grad output of the loss reduce stage from B step to the next F step.
    last_stage_forward_input_store = None
    last_stage_backward_input_store = None
    lm_head_reduce_output_store = None

    comm_wait_tensor = torch.Tensor([0]).cuda()
    comm_wait_tensor.record_stream(LM_HEAD_RES_REDUCE_STREAM)

    def broadcast_lm_head_input(microbatch_id, output_tensor, grad_output):
        """
        Assumes `output_tensor` is retrieved from `last_stage_forward_input_store`.
        We do not store it into `last_stage_forward_input_store` again.
        """
        nonlocal config, last_stage_backward_input_store, num_microbatches
        assert parallel_state.is_pipeline_last_stage(), \
            "lm head input must be broadcasted from the last stage"
        assert not config.variable_seq_lengths, 'not supported yet'
        if microbatch_id == 0:
            broadcast_tensor = output_tensor[0].to(dtype=torch.float32)
        elif microbatch_id == num_microbatches:
            broadcast_tensor = grad_output[0].unsqueeze(-1)
        else:
            broadcast_tensor = torch.cat([output_tensor[0].to(dtype=torch.float32), \
                                          grad_output[0].unsqueeze(-1)], -1)

        torch.distributed.broadcast(
            broadcast_tensor,
            parallel_state.get_pipeline_model_parallel_last_rank(),
            group=parallel_state.get_lm_head_model_parallel_group(),
            async_op=True,
        )

        if microbatch_id > 0:
            last_stage_backward_input_store = grad_output[0]

    def receive_lm_head_input(microbatch_id):
        nonlocal config, num_microbatches, last_stage_forward_input_store, \
                 last_stage_backward_input_store, lm_head_tensor_shapes, \
                 lm_head_reduce_output_store

        if not parallel_state.is_pipeline_last_stage():
            last_dim_shape = 0
            if microbatch_id < num_microbatches:
                last_dim_shape += lm_head_tensor_shapes[0][-1]
            if microbatch_id > 0:
                last_dim_shape += 1

            broadcast_tensor = torch.empty(
                lm_head_tensor_shapes[0][:-1] + (last_dim_shape,),
                dtype=torch.float32,
                device=torch.cuda.current_device(),
                requires_grad=True,
            )

            handle = torch.distributed.broadcast(
                broadcast_tensor,
                parallel_state.get_pipeline_model_parallel_last_rank(),
                group=parallel_state.get_lm_head_model_parallel_group(),
                async_op=True,
            )

        def callback():
            nonlocal broadcast_tensor, handle, microbatch_id, num_microbatches, \
                     config, last_stage_forward_input_store, last_stage_backward_input_store, \
                     lm_head_tensor_shapes, lm_head_reduce_output_store
            
            if not parallel_state.is_pipeline_last_stage():
                handle.wait()

            if microbatch_id < num_microbatches:
                if parallel_state.is_pipeline_last_stage():
                    output_tensor = last_stage_forward_input_store
                    last_stage_forward_input_store = None
                else:
                    output_tensor = broadcast_tensor[:, :, :lm_head_tensor_shapes[0][-1]].clone().to(dtype=config.pipeline_dtype)
            else:
                output_tensor = None
            
            if microbatch_id > 0:
                # Ensure that the reduction is complete.
                global LM_HEAD_RES_REDUCE_STREAM
                torch.cuda.current_stream().wait_stream(LM_HEAD_RES_REDUCE_STREAM)
                logits_max, sum_exp_logits, _, _ = lm_head_reduce_output_store

                if parallel_state.is_pipeline_last_stage():
                    grad_output = last_stage_backward_input_store
                    last_stage_backward_input_store = None
                else:
                    grad_output = broadcast_tensor[:, :, -1]

                if config.sequence_parallel:
                    gathered_tensor_shape = list(sum_exp_logits.shape)
                    gathered_tensor_shape[0] *= parallel_state.get_tensor_model_parallel_world_size()
                    logits_max_buffer = torch.empty(
                        gathered_tensor_shape,
                        dtype=logits_max.dtype,
                        device=torch.cuda.current_device(),
                    )
                    torch.distributed.all_gather_into_tensor(
                        logits_max_buffer,
                        logits_max.contiguous(),
                        group=parallel_state.get_tensor_model_parallel_group(),
                    )
                    logits_max = logits_max_buffer
                    sum_exp_logits_buffer = torch.empty(
                        gathered_tensor_shape,
                        dtype=sum_exp_logits.dtype,
                        device=torch.cuda.current_device(),
                    )
                    torch.distributed.all_gather_into_tensor(
                        sum_exp_logits_buffer,
                        sum_exp_logits.contiguous(),
                        group=parallel_state.get_tensor_model_parallel_group(),
                    )
                    sum_exp_logits = sum_exp_logits_buffer
                    grad_output_buffer = torch.empty(
                        gathered_tensor_shape,
                        dtype=grad_output.dtype,
                        device=torch.cuda.current_device(),
                    )
                    torch.distributed.all_gather_into_tensor(
                        grad_output_buffer,
                        grad_output.contiguous(),
                        group=parallel_state.get_tensor_model_parallel_group(),
                    )
                    grad_output = grad_output_buffer
            else:
                logits_max = None
                sum_exp_logits = None
                grad_output = None

            return [output_tensor], sum_exp_logits, logits_max, [grad_output]
        
        return callback

    def sequence_shard(t: torch.Tensor, *, dim: int = 0):
        nonlocal config
        if not config.sequence_parallel:
            return t
        world_size = parallel_state.get_tensor_model_parallel_world_size()
        rank = parallel_state.get_tensor_model_parallel_rank()
        dim_size = t.size(dim=dim) // world_size
        slices = [slice(None)] * t.dim()
        slices[dim] = slice(rank * dim_size, (rank + 1) * dim_size)
        return t[tuple(slices)]

    def reduce_lm_head_res(microbatch_id, logits_max, sum_exp_logits, predicted_logits, target_mask, grad_input):
        """
        Reduces `logits_max`, `sum_exp_logits`, `predicted_logits` and
        `grad_input` among all pipeline parallel ranks.
        """
        global LM_HEAD_RES_REDUCE_STREAM

        if microbatch_id < num_microbatches:
            logits_max = sequence_shard(logits_max)
            sum_exp_logits = sequence_shard(sum_exp_logits)
            predicted_logits = sequence_shard(predicted_logits)
            target_mask = sequence_shard(target_mask)

            for tensor in (logits_max, sum_exp_logits, predicted_logits, target_mask):
                tensor.record_stream(LM_HEAD_RES_REDUCE_STREAM)

            LM_HEAD_RES_REDUCE_STREAM.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(LM_HEAD_RES_REDUCE_STREAM):
                local_logits_max = logits_max.clone()
                handle = torch.distributed.all_reduce(
                    logits_max,
                    torch.distributed.ReduceOp.MAX,
                    group=parallel_state.get_lm_head_model_parallel_group(),
                    async_op=True,
                )
                handle.wait()
                local_logits_max -= logits_max

                predicted_logits += local_logits_max
                predicted_logits[target_mask] = 0.0
                handle = torch.distributed.all_reduce(
                    predicted_logits,
                    torch.distributed.ReduceOp.SUM,
                    group=parallel_state.get_lm_head_model_parallel_group(),
                    async_op=True,
                )
                handle.wait()

                local_logits_max.exp_()
                sum_exp_logits.mul_(local_logits_max)
                handle = torch.distributed.all_reduce(
                    sum_exp_logits,
                    torch.distributed.ReduceOp.SUM,
                    group=parallel_state.get_lm_head_model_parallel_group(),
                    async_op=True,
                )
                handle.wait()
        
        if microbatch_id > 0:
            grad_input.record_stream(LM_HEAD_RES_REDUCE_STREAM)
            LM_HEAD_RES_REDUCE_STREAM.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(LM_HEAD_RES_REDUCE_STREAM):
                handle = torch.distributed.all_reduce(
                    grad_input,
                    torch.distributed.ReduceOp.SUM,
                    group=parallel_state.get_lm_head_model_parallel_group(),
                    async_op=True,
                )
                handle.wait()

        return logits_max, sum_exp_logits, predicted_logits, grad_input

    def forward_step_helper(
        microbatch_id,
        input_tensor,
        run_timer
    ):
        """
        Executes forward step and completes language model head communication (if any). Returns
        the output tensor.

        Note: This function does not push the input and output tensors into `input_tensors` and
        `output_tensors`. The caller should do this after sending the output tensor.
        """
        nonlocal forward_step_func, data_iterator, model, num_microbatches, forward_data_store, \
                 config, collect_non_loss_data, encoder_decoder_xattn, total_num_tokens, forward_only, \
                 first_val_step, forward_only
        
        if get_args().profile:
            torch.cuda.nvtx.range_push(f"F{microbatch_id}")
        
        parallel_state.set_virtual_vocab_parallel_chunk(0)

        if parallel_state.is_pipeline_first_stage():
            input_tensor = [None]

        output_tensor, num_tokens = forward_step(
            forward_step_func,
            data_iterator,
            model[0],
            num_microbatches,
            input_tensor,
            forward_data_store,
            config,
            collect_non_loss_data,
            None,
            check_first_val_step(first_val_step, forward_only, microbatch_id == 0),
            current_microbatch=microbatch_id,
            encoder_decoder_xattn=encoder_decoder_xattn,
            skip_loss_compute=True,
            run_timer=run_timer
        )

        total_num_tokens += num_tokens.item()

        if parallel_state.is_pipeline_last_stage():
            nonlocal last_stage_forward_input_store
            last_stage_forward_input_store = output_tensor[0].clone().detach() \
                                             .to(config.pipeline_dtype).requires_grad_(True)
            if microbatch_id == 0:
                broadcast_lm_head_input(microbatch_id, [last_stage_forward_input_store], None)
        
        if get_args().profile:
            torch.cuda.nvtx.range_pop()

        return output_tensor
    
    def loss_calculation_helper(
        microbatch_id,
    ):
        assert parallel_state.is_pipeline_last_stage(), 'loss is only calculated at' \
            'the last pipeline parallel stage'
        nonlocal lm_head_reduce_output_store, num_microbatches, config, \
                 model_type, forward_step_func, data_iterator, model, forward_data_store, \
                 collect_non_loss_data, encoder_decoder_xattn, lm_head_reduce_output_store, \
                 first_val_step, forward_only, rank
        
        # Ensure that the reduction is complete.
        global LM_HEAD_RES_REDUCE_STREAM
        torch.cuda.current_stream().wait_stream(LM_HEAD_RES_REDUCE_STREAM)

        _, sum_exp_logits, predicted_logits, _ = lm_head_reduce_output_store

        # Calculate the loss. Then, execute the function that reduces the losses.

        input_tensor = torch.log(sum_exp_logits) - predicted_logits

        if config.sequence_parallel:
            gathered_tensor_shapes = list(lm_head_tensor_shapes[0][:-1])
            gathered_tensor_shapes[0] *= parallel_state.get_tensor_model_parallel_world_size()
            input_tensor_buffer = torch.empty(
                gathered_tensor_shapes,
                dtype=input_tensor.dtype,
                device=torch.cuda.current_device()
            )
            torch.distributed.all_gather_into_tensor(
                input_tensor_buffer,
                input_tensor,
                group=parallel_state.get_tensor_model_parallel_group(),
            )
            input_tensor = input_tensor_buffer

        input_tensor = [input_tensor.clone().detach().requires_grad_(True)]

        parallel_state.set_virtual_vocab_parallel_chunk(3)

        output_tensor, _ = forward_step(
            forward_step_func,
            data_iterator,
            model[3],
            num_microbatches,
            input_tensor,
            forward_data_store,
            config,
            collect_non_loss_data,
            None,
            check_first_val_step(first_val_step, forward_only, microbatch_id == 0),
            current_microbatch=microbatch_id,
            encoder_decoder_xattn=encoder_decoder_xattn,
            run_timer=False
        )

        output_tensor_grad = backward_step(
            input_tensor, output_tensor, [None], model_type, config,
            run_timer=False
        )

        if microbatch_id < num_microbatches:
            nonlocal last_stage_forward_input_store
            broadcast_lm_head_input(microbatch_id + 1, [last_stage_forward_input_store],
                                    [sequence_shard(output_tensor_grad[0])])
        else:
            broadcast_lm_head_input(microbatch_id + 1, None,
                                    [sequence_shard(output_tensor_grad[0])])

    input_embedding_backward_callback = lambda: None

    def backward_step_helper(
        microbatch_id,
        output_tensor_grad,
        run_timer,
    ):
        nonlocal input_tensors, output_tensors, num_microbatches, config, rank, enable_grad_sync, \
                 model_type, lm_head_reduce_output_store

        if get_args().profile:
            torch.cuda.nvtx.range_push(f"B{microbatch_id}")

        # Enable grad sync for the last microbatch in the batch if the full
        # backward pass completes in the 1F1B stage.
        if microbatch_id == num_microbatches - 1:
            if config.grad_sync_func is None or rank == 0:
                enable_grad_sync()

        if parallel_state.is_pipeline_last_stage():
            # Ensure that the reduction is complete.
            global LM_HEAD_RES_REDUCE_STREAM
            torch.cuda.current_stream().wait_stream(LM_HEAD_RES_REDUCE_STREAM)

            _, _, _, grad_input = lm_head_reduce_output_store

            # Calculate the input grads of the lm head layer, without calling backward.
            output_tensor_grad = [grad_input]

        input_tensor = input_tensors[0].pop(0)
        output_tensor = output_tensors[0].pop(0)

        parallel_state.set_virtual_vocab_parallel_chunk(0)

        input_tensor_grad = backward_step(
            input_tensor, output_tensor, output_tensor_grad, model_type, config,
            run_timer=run_timer
        )

        if parallel_state.is_pipeline_first_stage():
            EmbeddingStore.backward_store(input_tensor_grad[0])

        if get_args().profile:
            torch.cuda.nvtx.range_pop()

        return input_tensor_grad

    def lm_head_step_helper(
        microbatch_id,
        lm_head_inputs,
        run_timer
    ):
        nonlocal input_tensors, output_tensors, model_type, config, num_microbatches, \
                 forward_step_func, data_iterator, model, forward_data_store, \
                 collect_non_loss_data, encoder_decoder_xattn, first_val_step, forward_only

        if get_args().profile:
            torch.cuda.nvtx.range_push(f"S{microbatch_id}")

        lm_head_input_tensor, sum_exp_logits, logits_max, grad_output = lm_head_inputs

        parallel_state.set_virtual_vocab_parallel_chunk(1)
        CrossEntropyStore.microbatch_id = microbatch_id

        if (run_timer) and (0 < microbatch_id < num_microbatches):
            ScheduleTimers.for_chunk(0).s_cnt += 1
            ScheduleTimers.for_chunk(0).s.start()

        if microbatch_id > 0:
            input_tensor = input_tensors[1].pop(0)
            output_tensor = output_tensors[1].pop(0)

            # Only for weight grad updates, input grad returned is ignored.
            CrossEntropyStore.backward_store(sum_exp_logits, logits_max, grad_output[0])
            grad_input = backward_step(
                input_tensor, output_tensor, [grad_output[0].transpose(0, 1)], model_type, config,
                run_timer=False
            )
        else:
            grad_input = [None]

        if microbatch_id < num_microbatches:
            output_tensor, _ = forward_step(
                forward_step_func,
                data_iterator,
                model[1],
                num_microbatches,
                lm_head_input_tensor,
                forward_data_store,
                config,
                collect_non_loss_data,
                None,
                check_first_val_step(first_val_step, forward_only, microbatch_id == 0),
                current_microbatch=microbatch_id,
                encoder_decoder_xattn=encoder_decoder_xattn,
                skip_loss_compute=True,
                run_timer=False
            )
            output_tensor = [output_tensor[0].clone()]
            sum_exp_logits, logits_max, predicted_logits, target_mask, _, _ = \
                CrossEntropyStore.forward_get()

            input_tensors[1].append(lm_head_input_tensor)
            output_tensors[1].append(output_tensor)
            deallocate_output_tensor(output_tensor[0], config.deallocate_pipeline_outputs)

            lm_head_res = (logits_max, sum_exp_logits, predicted_logits, target_mask, grad_input[0])
        else:
            lm_head_res = (None, None, None, None, grad_input[0])
        
        if (run_timer) and (0 < microbatch_id < num_microbatches):
            ScheduleTimers.for_chunk(0).s.stop()

        if get_args().profile:
            torch.cuda.nvtx.range_pop()

        return lm_head_res

    input_embedding_output_shape = None
    
    def input_embedding_forward_step_helper(
        microbatch_id,
    ):
        nonlocal forward_step_func, data_iterator, model, num_microbatches, forward_data_store, \
                 config, collect_non_loss_data, encoder_decoder_xattn, forward_only, first_val_step

        parallel_state.set_virtual_vocab_parallel_chunk(2)

        input_tensor = [None]

        if get_args().profile:
            torch.cuda.nvtx.range_push(f"IF{microbatch_id}")

        if run_timer:
            ScheduleTimers.for_chunk(0).input_f_cnt += 1
            ScheduleTimers.for_chunk(0).input_f.start()

        output_tensor, _ = forward_step(
            forward_step_func,
            data_iterator,
            model[2],
            num_microbatches,
            input_tensor,
            forward_data_store,
            config,
            collect_non_loss_data,
            None,
            check_first_val_step(first_val_step, forward_only, microbatch_id == 0),
            current_microbatch=microbatch_id,
            encoder_decoder_xattn=encoder_decoder_xattn,
            skip_loss_compute=True,
            run_timer=False
        )

        if run_timer:
            ScheduleTimers.for_chunk(0).input_f.stop()
        
        if get_args().profile:
            torch.cuda.nvtx.range_pop()

        nonlocal input_embedding_output_shape
        input_embedding_output_shape = output_tensor[0].shape

        reduced_output_tensor = output_tensor[0].clone().detach().to(dtype=config.pipeline_dtype).requires_grad_(True)

        input_tensors[2].append(input_tensor)
        output_tensors[2].append(output_tensor)
        deallocate_output_tensor(output_tensor[0], config.deallocate_pipeline_outputs)

        def callback():
            nonlocal reduced_output_tensor

            torch.distributed.all_reduce(
                comm_wait_tensor,
                torch.distributed.ReduceOp.MAX,
                group=parallel_state.get_lm_head_model_parallel_group(),
                async_op=True,
            )

            handle = torch.distributed.all_reduce(
                reduced_output_tensor,
                torch.distributed.ReduceOp.SUM,
                group=parallel_state.get_lm_head_model_parallel_group(),
                async_op=True,
            )

            if parallel_state.is_pipeline_first_stage():
                EmbeddingStore.forward_store(reduced_output_tensor, handle)

            return
        
        return callback
    
    def input_embedding_backward_step_helper(
        microbatch_id
    ):
        parallel_state.set_virtual_vocab_parallel_chunk(2)

        input_tensor = input_tensors[2].pop(0)
        output_tensor = output_tensors[2].pop(0)

        if parallel_state.is_pipeline_first_stage():
            output_tensor_grad = [EmbeddingStore.backward_get()]
        else:
            output_tensor_grad = [
                torch.empty(
                    input_embedding_output_shape,
                    dtype=config.pipeline_dtype,
                    device=torch.cuda.current_device(),
                )
            ]

        torch.distributed.all_reduce(
                comm_wait_tensor,
                torch.distributed.ReduceOp.MAX,
                group=parallel_state.get_lm_head_model_parallel_group(),
                async_op=True,
            )

        handle = torch.distributed.broadcast(
            output_tensor_grad[0],
            parallel_state.get_pipeline_model_parallel_first_rank(),
            group=parallel_state.get_lm_head_model_parallel_group(),
            async_op=True,
        )

        def callback():
            nonlocal input_tensor, output_tensor,output_tensor_grad, model_type, \
                     config, handle
            
            handle.wait()

            if get_args().profile:
                torch.cuda.nvtx.range_push(f"IB{microbatch_id}")

            if run_timer:
                ScheduleTimers.for_chunk(0).input_b_cnt += 1
                ScheduleTimers.for_chunk(0).input_b.start()

            backward_step(
                input_tensor, output_tensor, output_tensor_grad, model_type, config,
                run_timer=False
            )

            if run_timer:
                ScheduleTimers.for_chunk(0).input_b.stop()

            if get_args().profile:
                torch.cuda.nvtx.range_pop()
        
        return callback

    assert not forward_only, "Not supported"

    num_input_embedding_forward_steps_remaining = num_microbatches
    num_input_embedding_backward_steps_remaining = num_microbatches

    for i in range(first_stage_num_warmup_microbatches - num_warmup_microbatches + 1):
        input_embedding_forward_step_helper(i)()
        num_input_embedding_forward_steps_remaining -= 1

    # Run warmup forward passes.
    for i in range(num_warmup_microbatches):
        # Decide to checkpoint all layers' activations of the current micro-batch
        input_tensor = recv_forward(recv_tensor_shapes, config)

        input_embedding_forward_step_helper(
            num_microbatches - num_input_embedding_forward_steps_remaining
        )()
        num_input_embedding_forward_steps_remaining -= 1
        output_tensor = forward_step_helper(
            i,
            input_tensor,
            run_timer,
        )
        # The communication for the last stage should be deferred until after the first S pass.
        if i < num_warmup_microbatches - 1:
            send_forward(output_tensor, send_tensor_shapes, config)
            input_tensors[0].append(input_tensor)
            output_tensors[0].append(output_tensor)
            deallocate_output_tensor(output_tensor[0], config.deallocate_pipeline_outputs)

    num_remaining_s_pass = num_microbatches
    lm_head_inputs = receive_lm_head_input(0)()
    lm_head_res = lm_head_step_helper(0, lm_head_inputs, run_timer)
    num_remaining_s_pass -= 1

    lm_head_res = reduce_lm_head_res(0, *lm_head_res)
    lm_head_reduce_output_store = lm_head_res

    input_embedding_forward_step_helper(
        num_microbatches - num_input_embedding_forward_steps_remaining
    )()
    num_input_embedding_forward_steps_remaining -= 1

    if num_warmup_microbatches > 0:
        send_forward(output_tensor, send_tensor_shapes, config)
        input_tensors[0].append(input_tensor)
        output_tensors[0].append(output_tensor)
        deallocate_output_tensor(output_tensor[0], config.deallocate_pipeline_outputs)

    if num_warmup_microbatches + 1 <= num_microbatches:
        # Decide to checkpoint all layers' activations of the current micro-batch
        input_tensor = recv_forward(recv_tensor_shapes, config)

    if num_warmup_microbatches + 1 <= num_microbatches:
        output_tensor = forward_step_helper(
            num_warmup_microbatches,
            input_tensor,
            run_timer,
        )
    
    if parallel_state.is_pipeline_last_stage():
        loss_calculation_helper(0)

    lm_head_inputs = receive_lm_head_input(1)()
    lm_head_res = lm_head_step_helper(1, lm_head_inputs, run_timer)
    num_remaining_s_pass -= 1

    lm_head_res = reduce_lm_head_res(1, *lm_head_res)
    lm_head_reduce_output_store = lm_head_res
    
    if num_warmup_microbatches + 1 <= num_microbatches:
        send_forward(output_tensor, send_tensor_shapes, config)
        input_tensors[0].append(input_tensor)
        output_tensors[0].append(output_tensor)
        deallocate_output_tensor(output_tensor[0], config.deallocate_pipeline_outputs)
    
    if num_warmup_microbatches + 2 <= num_microbatches:
        input_tensor = recv_forward(recv_tensor_shapes, config)

    if num_warmup_microbatches + 2 <= num_microbatches:
        output_tensor = forward_step_helper(
            num_warmup_microbatches + 1,
            input_tensor,
            run_timer,
        )
        if parallel_state.is_pipeline_last_stage():
            output_tensor_grad = send_forward_recv_backward(
                output_tensor, send_tensor_shapes, config
            )
            input_tensors[0].append(input_tensor)
            output_tensors[0].append(output_tensor)
            deallocate_output_tensor(output_tensor[0], config.deallocate_pipeline_outputs)

    num_warmup_s_pass_rank = num_warmup_s_pass[
        parallel_state.get_pipeline_model_parallel_rank()
    ]
    
    for i in range(num_warmup_s_pass_rank):
        if num_remaining_s_pass >= 0:
            lm_head_inputs = receive_lm_head_input(i + 2)()
            lm_head_res = lm_head_step_helper(i + 2, lm_head_inputs, run_timer)
            if (i + 2 >= num_warmup_s_pass[0] + 1) and (num_input_embedding_forward_steps_remaining > 0):
                input_embedding_forward_callback = input_embedding_forward_step_helper(
                    num_microbatches - num_input_embedding_forward_steps_remaining
                )
                num_input_embedding_forward_steps_remaining -= 1
            else:
                input_embedding_forward_callback = lambda: None
            if (
                (parallel_state.get_pipeline_model_parallel_rank() == parallel_state.get_pipeline_model_parallel_world_size() - 2)
                or (not is_bsf[parallel_state.get_pipeline_model_parallel_rank() + 1])
            ):
                input_embedding_forward_callback()
                lm_head_reduce_output_store = reduce_lm_head_res(i + 2, *lm_head_res)
            if i == num_warmup_s_pass_rank - 1:
                output_tensor_grad = send_forward_recv_backward(
                    output_tensor, send_tensor_shapes, config
                )
                input_tensors[0].append(input_tensor)
                output_tensors[0].append(output_tensor)
                deallocate_output_tensor(output_tensor[0], config.deallocate_pipeline_outputs)
            if (
                (parallel_state.get_pipeline_model_parallel_rank() != parallel_state.get_pipeline_model_parallel_world_size() - 2)
                and (is_bsf[parallel_state.get_pipeline_model_parallel_rank() + 1])
            ):
                input_embedding_forward_callback()
                lm_head_reduce_output_store = reduce_lm_head_res(i + 2, *lm_head_res)
            num_remaining_s_pass -= 1

    # Run 1F1B in steady state.
    for i in range(num_microbatches_remaining):
        if (
            (is_bsf[parallel_state.get_pipeline_model_parallel_rank()])
            or (offsets[parallel_state.get_pipeline_model_parallel_rank()] - f > -b)
        ):
            if num_microbatches - num_remaining_s_pass >= num_warmup_s_pass[0] + 3:
                input_embedding_backward_callback = input_embedding_backward_step_helper(
                    num_microbatches - num_input_embedding_backward_steps_remaining
                )
                num_input_embedding_backward_steps_remaining -= 1
            else:
                input_embedding_backward_callback = lambda: None
            receive_lm_head_input_callback = receive_lm_head_input(num_microbatches - num_remaining_s_pass)
        
        if parallel_state.is_pipeline_last_stage():
            loss_calculation_helper(i + 1)

        input_tensor_grad = backward_step_helper(
            i, output_tensor_grad, run_timer,
        )

        if is_bsf[parallel_state.get_pipeline_model_parallel_rank()]:
            lm_head_inputs = receive_lm_head_input_callback()
            lm_head_res = lm_head_step_helper(num_microbatches - num_remaining_s_pass, lm_head_inputs, run_timer)
            if (
                (num_microbatches - num_remaining_s_pass >= num_warmup_s_pass[0] + 1)
                and (num_input_embedding_forward_steps_remaining > 0)
            ):
                input_embedding_forward_callback = input_embedding_forward_step_helper(
                    num_microbatches - num_input_embedding_forward_steps_remaining
                )
                num_input_embedding_forward_steps_remaining -= 1
            else:
                input_embedding_forward_callback = lambda: None

            input_embedding_backward_callback()
            num_remaining_s_pass -= 1

        if not parallel_state.is_pipeline_last_stage():
            input_tensor = send_backward_recv_forward(
                input_tensor_grad, recv_tensor_shapes, config
            )

        if (
            (is_bsf[parallel_state.get_pipeline_model_parallel_rank()])
            and (offsets[parallel_state.get_pipeline_model_parallel_rank()] + f >= 0)
        ):
            input_embedding_forward_callback()
            lm_head_reduce_output_store = reduce_lm_head_res(num_microbatches - num_remaining_s_pass - 1, *lm_head_res)

        if parallel_state.is_pipeline_last_stage():
            input_tensor = send_backward_recv_forward(
                input_tensor_grad, recv_tensor_shapes, config
            )

        if (
            (not is_bsf[parallel_state.get_pipeline_model_parallel_rank()])
            and (offsets[parallel_state.get_pipeline_model_parallel_rank()] - f <= -b)
        ):
            if num_microbatches - num_remaining_s_pass >= num_warmup_s_pass[0] + 3:
                input_embedding_backward_callback = input_embedding_backward_step_helper(
                    num_microbatches - num_input_embedding_backward_steps_remaining
                )
                num_input_embedding_backward_steps_remaining -= 1
            else:
                input_embedding_backward_callback = lambda: None
            receive_lm_head_input_callback = receive_lm_head_input(num_microbatches - num_remaining_s_pass)

        output_tensor = forward_step_helper(
            i + num_warmup_microbatches + 2,
            input_tensor,
            run_timer,
        )

        if not is_bsf[parallel_state.get_pipeline_model_parallel_rank()]:
            lm_head_inputs = receive_lm_head_input_callback()
            lm_head_res = lm_head_step_helper(num_microbatches - num_remaining_s_pass, lm_head_inputs, run_timer)
            if (
                (num_microbatches - num_remaining_s_pass >= num_warmup_s_pass[0] + 1)
                and (num_input_embedding_forward_steps_remaining > 0)
            ):
                input_embedding_forward_callback = input_embedding_forward_step_helper(
                    num_microbatches - num_input_embedding_forward_steps_remaining
                )
                num_input_embedding_forward_steps_remaining -= 1
            else:
                input_embedding_forward_callback = lambda: None
            input_embedding_backward_callback()
            num_remaining_s_pass -= 1

        if (
            parallel_state.get_pipeline_model_parallel_rank()
            != parallel_state.get_pipeline_model_parallel_world_size() - 2
        ):
            output_tensor_grad = send_forward_recv_backward(
                output_tensor, send_tensor_shapes, config
            )

        if (
            (not is_bsf[parallel_state.get_pipeline_model_parallel_rank()])
            or (offsets[parallel_state.get_pipeline_model_parallel_rank()] + f < 0)
        ):
            input_embedding_forward_callback()
            lm_head_reduce_output_store = reduce_lm_head_res(num_microbatches - num_remaining_s_pass - 1, *lm_head_res)

        if (
            parallel_state.get_pipeline_model_parallel_rank()
            == parallel_state.get_pipeline_model_parallel_world_size() - 2
        ):
            output_tensor_grad = send_forward_recv_backward(
                output_tensor, send_tensor_shapes, config
            )

        input_tensors[0].append(input_tensor)
        output_tensors[0].append(output_tensor)
        deallocate_output_tensor(output_tensor[0], config.deallocate_pipeline_outputs)

    # Run cooldown backward passes.
    for i in range(num_microbatches - num_microbatches_remaining):        
        if (
            (is_bsf[parallel_state.get_pipeline_model_parallel_rank()]
             or (offsets[parallel_state.get_pipeline_model_parallel_rank()] - f > -b))
        ):
            if num_microbatches - num_remaining_s_pass >= num_warmup_s_pass[0] + 3:
                input_embedding_backward_callback = input_embedding_backward_step_helper(
                    num_microbatches - num_input_embedding_backward_steps_remaining
                )
                num_input_embedding_backward_steps_remaining -= 1
            else:
                input_embedding_backward_callback = lambda: None
            if num_remaining_s_pass >= 0:
                receive_lm_head_input_callback = receive_lm_head_input(num_microbatches - num_remaining_s_pass)

        # Enable async grad reduction in the last backward pass
        # Note: If grad sync function is provided, only enable
        # async grad reduction in first pipeline stage. Other
        # pipeline stages do grad reduction during pipeline
        # bubble.
        if i == num_microbatches - num_microbatches_remaining - 1:
            if config.grad_sync_func is None or rank == 0:
                enable_grad_sync()
        
        if parallel_state.is_pipeline_last_stage() and (i + num_microbatches_remaining + 1 < num_microbatches):
            loss_calculation_helper(i + num_microbatches_remaining + 1)

        input_tensor_grad = backward_step_helper(
            i + num_microbatches_remaining, output_tensor_grad, run_timer,
        )

        s_executed = False

        if is_bsf[parallel_state.get_pipeline_model_parallel_rank()]:
            if num_remaining_s_pass >= 0:
                lm_head_inputs = receive_lm_head_input_callback()
                lm_head_res = lm_head_step_helper(num_microbatches - num_remaining_s_pass, lm_head_inputs, run_timer)
                num_remaining_s_pass -= 1
                s_executed = True
            input_embedding_backward_callback()

        if not parallel_state.is_pipeline_last_stage():
            send_backward(input_tensor_grad, recv_tensor_shapes, config)

        if (
            s_executed
            and (offsets[parallel_state.get_pipeline_model_parallel_rank()] + f >= 0)
        ):
            lm_head_reduce_output_store = reduce_lm_head_res(num_microbatches - num_remaining_s_pass - 1, *lm_head_res)
            s_executed = False
        
        if parallel_state.is_pipeline_last_stage():
            send_backward(input_tensor_grad, recv_tensor_shapes, config)
        
        if (
            (not is_bsf[parallel_state.get_pipeline_model_parallel_rank()]
             and (offsets[parallel_state.get_pipeline_model_parallel_rank()] - f <= -b))
        ):
            if num_microbatches - num_remaining_s_pass >= num_warmup_s_pass[0] + 3:
                input_embedding_backward_callback = input_embedding_backward_step_helper(
                    num_microbatches - num_input_embedding_backward_steps_remaining
                )
                num_input_embedding_backward_steps_remaining -= 1
            else:
                input_embedding_backward_callback = lambda: None
            if num_remaining_s_pass >= 0:
                receive_lm_head_input_callback = receive_lm_head_input(num_microbatches - num_remaining_s_pass)

        if not is_bsf[parallel_state.get_pipeline_model_parallel_rank()]:
            if num_remaining_s_pass >= 0:
                lm_head_inputs = receive_lm_head_input_callback()
                lm_head_res = lm_head_step_helper(num_microbatches - num_remaining_s_pass, lm_head_inputs, run_timer)
                num_remaining_s_pass -= 1
                s_executed = True
            input_embedding_backward_callback()
        
        if (
            parallel_state.get_pipeline_model_parallel_rank()
            != parallel_state.get_pipeline_model_parallel_world_size() - 2
        ):
            if i + 1 < num_microbatches - num_microbatches_remaining:
                output_tensor_grad = recv_backward(
                    send_tensor_shapes, config
                )

        if s_executed:
            lm_head_reduce_output_store = reduce_lm_head_res(num_microbatches - num_remaining_s_pass - 1, *lm_head_res)
        
        if (
            parallel_state.get_pipeline_model_parallel_rank()
            == parallel_state.get_pipeline_model_parallel_world_size() - 2
        ):
            if i + 1 < num_microbatches - num_microbatches_remaining:
                output_tensor_grad = recv_backward(
                    send_tensor_shapes, config
                )
    
    while num_input_embedding_backward_steps_remaining > 0:
        input_embedding_backward_step_helper(
            num_microbatches - num_input_embedding_backward_steps_remaining
        )()
        num_input_embedding_backward_steps_remaining -= 1

    # Launch any remaining grad reductions.
    if no_sync_context is not None:
        enable_grad_sync()
        if config.grad_sync_func is not None:
            config.grad_sync_func(model.parameters())

    if config.finalize_model_grads_func is not None and not forward_only:

        # If defer_embedding_wgrad_compute is enabled we need to do the
        # weight gradient GEMM's here.
        finish_embedding_wgrad_compute(config, embedding_module)

        # Finalize model grads (perform full grad all-reduce / reduce-scatter for
        # data parallelism, layernorm all-reduce for sequence parallelism, and
        # embedding all-reduce for pipeline parallelism).
        config.finalize_model_grads_func(
            model, total_num_tokens if config.calculate_per_token_loss else None
        )

    if config.timers is not None:
        config.timers('forward-backward').stop()

    return forward_data_store
