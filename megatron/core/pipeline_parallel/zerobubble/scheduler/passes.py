import dataclasses
from typing import List

from megatron.core.pipeline_parallel.zerobubble.scheduler.communication import run_communication_passes, \
    validate_communication
from megatron.core.pipeline_parallel.zerobubble.scheduler.graph import \
    GraphConfig, F, B, W, BW, IF, IB, S, ScheduledNode
from megatron.training import get_args


def run_schedule_passes(
    config: GraphConfig,
    local_order: List[List[ScheduledNode]],
    validate=True,
) -> List[List[ScheduledNode]]:
    pre_validate(local_order)
    local_order = add_send_recv_peer_stage(config, local_order)
    local_order = add_time(config, local_order)
    if get_args().enable_vocab_parallel:
        local_order, s_timings, if_timings, ib_timings = add_embedding_passes(config, local_order)
    else:
        s_timings, if_timings, ib_timings = None, None, None
    local_order = run_communication_passes(config, local_order, s_timings, if_timings, ib_timings)
    print_schedule(local_order)
    # if validate:
    #     validate_communication(local_order)
    return local_order


def add_embedding_passes(
    config: GraphConfig,
    local_order: List[List[ScheduledNode]],
) -> tuple[List[List[ScheduledNode]], List[int], List[int], List[int]]:
    # only works for zb-v schedule
    s_timings = []
    for event in local_order[0]:
        if (event.type == F) and (event.chunk == 1):
            s_timings.append(event.completion_time + sum(config.cost_b) / config.n_stages)
    s_timings.append(s_timings[-1] + sum(config.cost_b) / config.n_stages * 2 + sum(config.cost_f) / config.n_stages * 2)

    warmup_mbs = config.n_stages // 2 + 1

    if_timings = [0] * config.n_micro
    for i in range(config.n_micro):
        if_timings[i] = i - config.n_micro
    
    for i in range(warmup_mbs, config.n_micro):
        if_timings[i] = s_timings[i - warmup_mbs]

    max_completion_time = max([local_order[rank][-1].completion_time for rank in range(config.n_stages)])

    ib_timings = [0] * config.n_micro
    for i in range(config.n_micro):
        ib_timings[i] = max_completion_time + i

    for i in range(0, config.n_micro - warmup_mbs - 2):
        ib_timings[i] = s_timings[i + warmup_mbs + 2]

    for rank in range(config.n_stages):
        for i in range(config.n_micro):
            local_order[rank].append(ScheduledNode(
                type=IF,
                stage=rank,
                microbatch=i,
                start_time=if_timings[i],
                completion_time=if_timings[i],
            ))
        for i in range(config.n_micro + 1):
            local_order[rank].append(ScheduledNode(
                type=S,
                stage=rank,
                microbatch=i,
                start_time=s_timings[i],
                completion_time=s_timings[i],
            ))
        for i in range(config.n_micro):
            local_order[rank].append(ScheduledNode(
                type=IB,
                stage=rank,
                microbatch=i,
                start_time=ib_timings[i],
                completion_time=ib_timings[i],
            ))
    return local_order, s_timings, if_timings, ib_timings


def viz_node(node: ScheduledNode):
    short = True
    if short:
        name_map = {
            'F': 'F',
            'B': 'B',
            'W': 'W',
            'BW': 'B',
            'S': 'S',
            'IF': 'IF',
            'IB': 'IB',
            'SEND_FORWARD': 'SF',
            'RECV_FORWARD': 'RF',
            'SEND_BACKWARD': 'SB',
            'RECV_BACKWARD': 'RB',
            'REDUCE_INPUT_EMBD_FORWARD': 'SIF',
            'BROADCAST_INPUT_EMBD_BACKWARD': 'RIB',
            'BROADCAST_OUTPUT_EMBD': 'RS',
            'REDUCE_OUTPUT_EMBD': 'SS',
            'POST_VALIDATION': 'PV',
            'RECV_POST_VALIDATION': 'RPV',
            'SEND_POST_VALIDATION': 'SPV',
        }
        n = name_map[node.type.value]
        func = n.lower() if node.chunk == 0 else n.upper()
        return f"{func}{node.microbatch}"
    func = node.type.value.lower() if node.chunk == 0 else node.type.value.upper()
    recv = f"r{node.recv_peer_stage}" if node.recv_peer_stage is not None else ""
    send = f"s{node.send_peer_stage}" if node.send_peer_stage is not None else ""
    return f"{recv}{func}{node.microbatch}{send}"


def print_schedule(res):
    for nodes in res:
        ns = ' '.join(map(viz_node, nodes))
        print(ns)


def pre_validate(local_order: List[List[ScheduledNode]]):
    for stage, nodes in enumerate(local_order):
        for n in nodes:
            assert stage == n.stage, f"stage not correct: expected stage: {stage} node: {n}"


def add_send_recv_peer_stage(
    config: GraphConfig,
    local_order: List[List[ScheduledNode]],
) -> List[List[ScheduledNode]]:
    nodes = sum(local_order, [])
    node_map = {node.get_key(): node for node in nodes}

    for n in nodes:
        prev_key = n.get_prev_key(config.num_layer_groups())
        if prev_key is None:
            continue
        if prev_key not in node_map:
            raise ValueError(f"Cannot find prev node {n.get_key()} depends on {prev_key}")
        prev_node = node_map[prev_key]
        if prev_node.stage == n.stage:
            continue
        prev_node.send_peer_stage = n.stage
        n.recv_peer_stage = prev_node.stage
    return local_order


def add_time(config: GraphConfig, local_order: List[List[ScheduledNode]]) -> List[List[ScheduledNode]]:
    nodes = sum(local_order, [])
    node_map = {node.get_key(): node for node in nodes}

    type_cost = {
        F: config.cost_f,
        B: config.cost_b,
        W: config.cost_w,
        BW: config.cost_b + config.cost_w,
    }
    new_local_order = [[] for _ in local_order]
    completion_time = {}
    stage_curr_t = [0.0 for _ in local_order]

    stage_curr_index = [0 for _ in local_order]
    ct = 0
    while ct < len(nodes):
        found = False
        pending = []
        for stage in range(len(local_order)):
            if stage_curr_index[stage] >= len(local_order[stage]):
                continue
            node = local_order[stage][stage_curr_index[stage]]
            prev_compute_node = node.get_prev_key(config.num_layer_groups())
            if prev_compute_node is not None and prev_compute_node not in completion_time:
                pending.append((node.get_key(), prev_compute_node))
                continue
            stage_curr_index[stage] += 1
            t = stage_curr_t[stage]
            if prev_compute_node is not None:
                prev_t = completion_time[prev_compute_node]
                if node_map[prev_compute_node].stage != node.stage:
                    prev_t += config.cost_comm
                t = max(prev_t, t)
            compute_t = type_cost[node.type][node.stage]
            end_t = t + compute_t
            completion_time[node.get_key()] = end_t
            stage_curr_t[node.stage] = end_t

            new_local_order[node.stage].append(
                dataclasses.replace(
                    dataclasses.replace(node, completion_time=end_t),
                    start_time=t,
                )
            )
            found = True
            ct += 1
        if not found:
            print(f"ERROR: can't find next runnable node. stage_curr_index: {stage_curr_index} completed {completion_time}")
            for (node, prev) in pending:
                print(f"ERROR: Pending {node} {prev}")
            raise RuntimeError(f"Cannot find next runnable node.")

    assert len(new_local_order) == len(local_order)
    for new, prev in zip(new_local_order, local_order):
        assert len(new) == len(prev)
    return new_local_order
