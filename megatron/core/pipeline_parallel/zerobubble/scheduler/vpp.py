from megatron.core.pipeline_parallel.zerobubble.scheduler.graph import GraphConfig, ScheduledNode, F, BW


def create_schedule(config: GraphConfig):
    num_microbatches = config.n_micro
    pipeline_parallel_size = config.n_stages
    num_model_chunks = config.max_chunks
    total_num_microbatches = num_microbatches * num_model_chunks

    if num_microbatches == 1:
        # This case is mainly for debugging
        return create_schedule_with_one_mb(config)

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
            layer_group_idx = config.n_stages * chunk + stage
            order.append(
                ScheduledNode(
                    type=func_type,
                    stage=stage,
                    microbatch=mb,
                    chunk=chunk,
                    layer_group_idx=layer_group_idx,
                )
            )
        local_order.append(order)
    return local_order


def create_schedule_with_one_mb(config: GraphConfig):
    local_order = []
    for stage in range(config.n_stages):
        funcs = [
            (F, 0, 0),
            (F, 0, 1),
            (BW, 0, 1),
            (BW, 0, 0),
        ]
        # funcs.append((F, 0, 0))
        # funcs.append((F, 0, 1))
        # funcs.append((BW, 0, 1))
        # funcs.append((BW, 0, 0))
        order = []
        for func_type, mb, chunk in funcs:
            layer_group_idx = config.n_stages * chunk + stage
            order.append(
                ScheduledNode(
                    type=func_type,
                    stage=stage,
                    microbatch=mb,
                    chunk=chunk,
                    layer_group_idx=layer_group_idx,
                )
            )
        local_order.append(order)
    return local_order
