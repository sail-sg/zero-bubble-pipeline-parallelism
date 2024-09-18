from megatron.core.pipeline_parallel.zerobubble.scheduler.communication import comm_goes_down, comm_goes_up, last_stage, \
    next_stage
from megatron.core.pipeline_parallel.zerobubble.scheduler.graph import GraphConfig, ScheduledNode, F, BW


def create_schedule(config: GraphConfig):
    num_microbatches = config.n_micro
    pipeline_parallel_size = config.n_stages
    num_model_chunks = config.max_chunks
    total_num_microbatches = num_microbatches * num_model_chunks

    if num_microbatches % pipeline_parallel_size != 0:
        msg = f'number of microbatches ({num_microbatches}) is not divisible by '
        msg += f'pipeline-model-parallel-size ({pipeline_parallel_size}) '
        msg += 'when using interleaved schedule'
        raise RuntimeError(msg)

    def get_model_chunk_id(microbatch_id, forward):
        """Helper method to get the model chunk ID given the iteration number."""
        microbatch_id_in_group = microbatch_id % (pipeline_parallel_size * num_model_chunks)
        model_chunk_id = microbatch_id_in_group // pipeline_parallel_size
        if not forward:
            model_chunk_id = num_model_chunks - model_chunk_id - 1
        return model_chunk_id

    def get_microbatch_id_in_model_chunk(iteration_id):
        """Helper method to get the microbatch_id within model chunk given the iteration number."""
        iteration_group_id = iteration_id // (pipeline_parallel_size * num_model_chunks)
        microbatch_id_in_model_chunk = (iteration_group_id * pipeline_parallel_size) + (
            iteration_id % pipeline_parallel_size
        )
        return microbatch_id_in_model_chunk

    local_order = []
    for stage in range(config.n_stages):
        order = []

        num_warmup_microbatches = (pipeline_parallel_size - stage - 1) * 2
        num_warmup_microbatches += (num_model_chunks - 1) * pipeline_parallel_size
        num_warmup_microbatches = min(num_warmup_microbatches, total_num_microbatches)
        num_microbatches_remaining = total_num_microbatches - num_warmup_microbatches

        funcs = []
        for k in range(num_warmup_microbatches):
            chunk = get_model_chunk_id(k, forward=True)
            mb = get_microbatch_id_in_model_chunk(k)
            funcs.append((F, mb, chunk))

        for k in range(num_microbatches_remaining):
            # Forward
            forward_k = k + num_warmup_microbatches
            chunk = get_model_chunk_id(forward_k, forward=True)
            mb = get_microbatch_id_in_model_chunk(forward_k)
            funcs.append((F, mb, chunk))
            # Backward
            backward_k = k
            chunk = get_model_chunk_id(backward_k, forward=False)
            mb = get_microbatch_id_in_model_chunk(backward_k)
            funcs.append((BW, mb, chunk))

        for backward_k in range(num_microbatches_remaining, total_num_microbatches):
            chunk = get_model_chunk_id(backward_k, forward=False)
            mb = get_microbatch_id_in_model_chunk(backward_k)
            funcs.append((BW, mb, chunk))

        print(" ".join([f"{t.value}{mb}.{chunk}" for (t, mb, chunk) in funcs]))

        for func_type, mb, chunk in funcs:
            if func_type == F:
                recv_peer_stage = last_stage(stage, config.n_stages, wrap_around=chunk > 0)
                send_peer_stage = next_stage(stage, config.n_stages, wrap_around=chunk < config.max_chunks - 1)
                # recv_peer_stage, send_peer_stage = comm_goes_down(stage, config.n_stages, wrap_around=True)
            else:
                recv_peer_stage = next_stage(stage, config.n_stages, wrap_around=chunk < config.max_chunks - 1)
                send_peer_stage = last_stage(stage, config.n_stages, wrap_around=chunk > 0)
                # recv_peer_stage, send_peer_stage = comm_goes_up(stage, config.n_stages, wrap_around=True)
            layer_group_idx = config.n_stages * chunk + stage
            order.append(
                ScheduledNode(
                    type=func_type,
                    stage=stage,
                    microbatch=mb,
                    chunk=chunk,
                    layer_group_idx=layer_group_idx,
                    recv_peer_stage=recv_peer_stage,
                    send_peer_stage=send_peer_stage,
                )
            )
        local_order.append(order)
    return local_order
