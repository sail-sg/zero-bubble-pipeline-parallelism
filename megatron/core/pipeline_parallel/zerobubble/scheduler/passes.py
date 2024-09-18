import dataclasses
import itertools
from typing import List

from megatron.core.pipeline_parallel.zerobubble.scheduler.communication import run_communication_passes
from megatron.core.pipeline_parallel.zerobubble.scheduler.graph import GraphConfig, F, B, W, BW, ScheduledNode, NodeKey


def run_schedule_passes(
        config: GraphConfig,
        local_order: List[List[ScheduledNode]]) -> List[List[ScheduledNode]]:
    # local_order = merge_consecutive_bw(local_order)
    local_order = add_prev_compute_node(config, local_order)
    local_order = add_time(config, local_order)
    local_order = run_communication_passes(config, local_order)
    print_schedule(local_order)
    return local_order


def viz_node(node: ScheduledNode):
    func = node.type.value.lower() if node.chunk == 0 else node.type.value.upper()
    recv = f"r{node.recv_peer_stage}" if node.recv_peer_stage is not None else ""
    send = f"s{node.send_peer_stage}" if node.send_peer_stage is not None else ""
    return f"{recv}{func}{node.microbatch}{send}"


def print_schedule(res):
    for nodes in res:
        ns = ' '.join(map(viz_node, nodes))
        print(ns)


def merge_consecutive_bw(local_order: List[List[ScheduledNode]]):
    new_local_order = [[] for _ in local_order]
    for stage in range(len(local_order)):
        skip_next = False
        for curr, next in itertools.zip_longest(local_order[stage], local_order[stage][1:]):
            if skip_next:
                skip_next = False
                continue
            if curr.type == B and next and next.type == W \
                    and curr.microbatch == next.microbatch \
                    and curr.chunk == next.chunk \
                    and curr.seq_split_idx == next.seq_split_idx:
                new_local_order[stage].append(dataclasses.replace(curr, type=BW))
                skip_next = True
            else:
                new_local_order[stage].append(curr)
    return new_local_order


def add_prev_compute_node(config: GraphConfig, local_order: List[List[ScheduledNode]]) -> List[List[ScheduledNode]]:
    """Only for F and B. Ignore the dependencies between F and B"""
    # TODO: remove node_keys
    node_keys = set()
    node_map = {}
    for stage_nodes in local_order:
        for node in stage_nodes:
            node_keys.add(node.get_key())
            node_map[node.get_new_key()] = node

    new_local_order = [[] for _ in local_order]
    for stage in range(len(local_order)):
        for n in local_order[stage]:
            if n.type == W or n.recv_peer_stage is None:
                new_local_order[stage].append(n)
                continue
            prev_stage = n.recv_peer_stage
            prev_chunk = n.chunk
            prev_key = n.get_prev_key(config.num_layer_groups())
            if prev_key is not None:
                prev_chunk = node_map[prev_key].chunk
                # TODO: remove this check
                if n.type == F:
                    prev_chunk0 = n.chunk if stage - prev_stage == 1 else n.chunk - 1
                else:
                    prev_chunk0 = n.chunk if prev_stage - stage == 1 else n.chunk + 1
                assert prev_chunk == prev_chunk0
            assert 0 <= prev_chunk < config.max_chunks, f"Invalid prev_chunk {prev_chunk} from {n}"

            if n.type in (B, BW):
                # For B and BW, it's previous type can be different.
                for t in (B, BW):
                    prev_node = NodeKey(type=t, stage=prev_stage, minibatch=n.microbatch, chunk=prev_chunk, seq_split_idx=n.seq_split_idx)
                    if prev_node in node_keys:
                        break
            else:
                prev_node = NodeKey(type=n.type, stage=prev_stage, minibatch=n.microbatch, chunk=prev_chunk, seq_split_idx=n.seq_split_idx)
            if prev_node not in node_keys:
                raise ValueError(f"cannot find previous node for {n}")
            new_local_order[stage].append(dataclasses.replace(n, prev_compute_node=prev_node))

    assert len(new_local_order) == len(local_order)
    for new, prev in zip(new_local_order, local_order):
        assert len(new) == len(prev)
    return new_local_order


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
            if node.prev_compute_node is not None and node.prev_compute_node not in completion_time:
                pending.append((node.get_key(), node.prev_compute_node))
                continue
            stage_curr_index[stage] += 1
            t = stage_curr_t[stage]
            if node.prev_compute_node is not None:
                prev_t = completion_time[node.prev_compute_node]
                if node_map[node.prev_compute_node].stage != node.stage:
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
