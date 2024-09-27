from megatron.core.pipeline_parallel.zerobubble.scheduler.graph import GraphConfig, ScheduledNode, F, BW


def create_schedule(config: GraphConfig):
    local_order = []
    for stage in range(config.n_stages):
        order = []
        num_warmup = min(config.n_stages - stage - 1, config.n_micro)
        num_remaining = config.n_micro - num_warmup
        funcs = []
        for mb in range(num_warmup):
            funcs.append((F, mb))
        for mb in range(num_remaining):
            funcs.append((F, num_warmup + mb))
            funcs.append((BW, mb))
        for i in range(num_warmup):
            funcs.append((BW, num_remaining + i))

        print(" ".join([f"{t.value}{mb}" for (t, mb) in funcs]))

        for func_type, mb in funcs:
            order.append(
                ScheduledNode(
                    type=func_type,
                    stage=stage,
                    microbatch=mb,
                    layer_group_idx=stage,
                )
            )
        local_order.append(order)
    return local_order
