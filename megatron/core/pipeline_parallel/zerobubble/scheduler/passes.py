import dataclasses
import itertools
from typing import List

from megatron.core.pipeline_parallel.zerobubble.scheduler.communication import run_communication_passes
from .graph import GraphConfig, ScheduledNode


def run_schedule_passes(
        config: GraphConfig,
        local_order: List[List[ScheduledNode]]) -> List[List[ScheduledNode]]:
    local_order = merge_consecutive_bw(local_order)
    # print_schedule(local_order)
    local_order = run_communication_passes(config, local_order)
    return local_order


def viz_node(node: ScheduledNode):
    func = node.type.lower() if node.chunk == 0 else node.type.upper()
    recv = f"r{node.recv_peer_stage}" if node.recv_peer_stage is not None else ""
    send = f"s{node.send_peer_stage}" if node.send_peer_stage is not None else ""
    return f"{recv}{func}{node.minibatch}{send}"


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
            if curr.type == 'B' and next and next.type == 'W' \
                    and curr.minibatch == next.minibatch \
                    and curr.chunk == next.chunk:
                new_local_order[stage].append(dataclasses.replace(curr, type='BW'))
                skip_next = True
            else:
                new_local_order[stage].append(curr)
    return new_local_order
