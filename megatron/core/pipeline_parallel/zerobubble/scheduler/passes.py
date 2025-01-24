import dataclasses
from typing import List

from megatron.core.pipeline_parallel.zerobubble.scheduler.communication import run_communication_passes, \
    validate_communication, add_communication_nodes_without_sorting
from megatron.core.pipeline_parallel.zerobubble.scheduler.offloading import add_offload
from megatron.core.pipeline_parallel.zerobubble.scheduler.graph import GraphConfig, F, B, W, BW, R, ScheduledNode


def run_schedule_passes(
    config: GraphConfig,
    local_order: List[List[ScheduledNode]],
    post_validation=False,
    offload_time=None,
    validate=True,
) -> List[List[ScheduledNode]]:
    pre_validate(local_order)
    local_order = add_send_recv_peer_stage(config, local_order)
    local_order = add_time(config, local_order)
    if offload_time is not None:
        local_order = add_offload(config, local_order, offload_time)
    # TODO: Remove run_communication_passes once we confirm the new impl works.
    # local_order = run_communication_passes(config, local_order, post_validation)
    local_order = add_communication_nodes_without_sorting(config, local_order, post_validation)
    print_schedule(local_order)
    if validate:
        validate_communication(local_order)
    return local_order


def viz_node(node: ScheduledNode):
    short = True
    if short:
        name_map = {
            'F': 'F',
            'B': 'B',
            'W': 'W',
            'BW': 'B',
            'R': 'R',
            'SEND_FORWARD': 'SF',
            'RECV_FORWARD': 'RF',
            'SEND_BACKWARD': 'SB',
            'RECV_BACKWARD': 'RB',
            'POST_VALIDATION': 'PV',
            'RECV_POST_VALIDATION': 'RPV',
            'SEND_POST_VALIDATION': 'SPV',
            'OFFLOAD_SEND_START': 'OSS',
            'OFFLOAD_SEND_END': 'OSE',
            'OFFLOAD_RECV_PREP': 'ORP',
            'OFFLOAD_RECV_START': 'ORS',
            'OFFLOAD_RECV_END': 'ORE',
            'OFFLOAD_BARRIER': 'OB',
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
    if local_order[0][0].start_time is not None:
        return local_order

    nodes = sum(local_order, [])
    node_map = {node.get_key(): node for node in nodes}

    type_cost = {
        F: config.cost_f,
        B: config.cost_b,
        W: config.cost_w,
        BW: config.cost_b + config.cost_w,
        R: config.cost_f,
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
