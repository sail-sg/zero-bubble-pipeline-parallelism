from megatron.training import get_args

from megatron.core.pipeline_parallel.zerobubble.scheduler.graph import GraphConfig, ScheduledNode, F, BW


def create_schedule(config: GraphConfig):
    local_order = []
    for stage in range(config.n_stages):
        order = []
        num_sq = get_args().num_seq_splits
        num_warmup = min(config.n_stages - stage - 2 + num_sq, config.n_micro * num_sq)
        num_remaining = config.n_micro * num_sq - num_warmup
        funcs = []
        for i in range(num_warmup):
            mb = i // num_sq
            sq = i % num_sq
            funcs.append((F, mb, sq))
        for i in range(num_remaining):
            mb = (num_warmup + i) // num_sq
            sq = (num_warmup + i) % num_sq
            funcs.append((F, mb, sq))
            mb = i // num_sq
            sq = i % num_sq
            funcs.append((BW, mb, num_sq - 1 - sq))
        for i in range(num_remaining, num_remaining + num_warmup):
            mb = i // num_sq
            sq = num_sq - 1 - i % num_sq
            funcs.append((BW, mb, sq))

        print(" ".join([f"{t.value}{mb}-{sq}" for (t, mb, sq) in funcs]))

        for func_type, mb, sq in funcs:
            order.append(
                ScheduledNode(
                    type=func_type,
                    stage=stage,
                    microbatch=mb,
                    seq_split_idx=sq,
                    layer_group_idx=stage,
                )
            )
        local_order.append(order)
    return local_order
