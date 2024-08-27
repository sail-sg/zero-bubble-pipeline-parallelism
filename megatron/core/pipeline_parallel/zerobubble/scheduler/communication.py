import dataclasses
from typing import List

from megatron.core.pipeline_parallel.zerobubble.scheduler import ScheduledNode, CommDirection, NodeKey
from megatron.core.pipeline_parallel.zerobubble.scheduler.graph import GraphConfig, TYPE_TO_CAT


class CommSet:
    def __init__(self):
        self.comm_id = {}
        self.comm_id_counter = 0


def run_communication_passes(
        config: GraphConfig,
        local_order: List[List[ScheduledNode]]) -> List[List[ScheduledNode]]:
    comm_set = CommSet()
    local_order = add_prev_compute_node(local_order)
    local_order = add_time(config, local_order)
    local_order = add_post_validation_nodes(config, comm_set, local_order)
    local_order = add_communication_nodes(config, comm_set, local_order)
    local_order = reorder_communication(config, comm_set, local_order)
    local_order = tag_rollback_communication(config, local_order)
    return local_order


def add_prev_compute_node(local_order: List[List[ScheduledNode]]) -> List[List[ScheduledNode]]:
    """Only for F and B. Ignore the dependencies between F and B"""
    node_keys = set()
    for stage_nodes in local_order:
        for node in stage_nodes:
            node_keys.add(node.get_key())

    new_local_order = [[] for _ in local_order]
    for stage in range(len(local_order)):
        for n in local_order[stage]:
            if n.type == 'W' or n.recv_peer_stage is None:
                new_local_order[stage].append(n)
                continue
            prev_stage = n.recv_peer_stage
            if n.type in ('B', 'BW'):
                # For B and BW, it's previous type can be different.
                for t in ('B', 'BW'):
                    prev_node = NodeKey(type=t, stage=prev_stage, minibatch=n.microbatch, chunk=n.chunk)
                    if prev_node in node_keys:
                        break
            else:
                prev_node = NodeKey(type=n.type, stage=prev_stage, minibatch=n.microbatch, chunk=n.chunk)
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
        'F': config.cost_f,
        'B': config.cost_b,
        'W': config.cost_w,
        'BW': config.cost_b + config.cost_w,
    }
    new_local_order = [[] for _ in local_order]
    completion_time = {}
    stage_curr_t = [0.0 for _ in local_order]

    stage_curr_index = [0 for _ in local_order]
    ct = 0
    while ct < len(nodes):
        found = False
        for stage in range(len(local_order)):
            if stage_curr_index[stage] >= len(local_order[stage]):
                continue
            node = local_order[stage][stage_curr_index[stage]]
            if node.prev_compute_node is not None and node.prev_compute_node not in completion_time:
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
        assert found

    assert len(new_local_order) == len(local_order)
    for new, prev in zip(new_local_order, local_order):
        assert len(new) == len(prev)
    return new_local_order


def get_post_validation_time(config: GraphConfig, stage, local_order):
    node_map = {n.get_key(): n for n in sum(local_order, [])}
    deadline_idx = next(i for i, n in enumerate(local_order[stage]) if n.type != 'F' or n.chunk != 0)
    pv_id = min(2 * (config.n_stages - 1 - stage), config.n_micro - 1)
    pv_id = min(pv_id, deadline_idx - 1)
    end_time = node_map[NodeKey('F', chunk=0, stage=stage, minibatch=pv_id)].completion_time
    cat = TYPE_TO_CAT[local_order[stage][pv_id].type]
    cost = config.get_cost(stage, cat)
    return end_time - cost - config.cost_comm


def add_post_validation_nodes(
        config: GraphConfig,
        comm_set: CommSet,
        local_order: List[List[ScheduledNode]]) -> List[List[ScheduledNode]]:
    assert len(local_order) == config.n_stages

    post_validation_time = 0
    for stage in range(config.n_stages - 1, -1, -1):
        pv_time = get_post_validation_time(config, stage, local_order)
        post_validation_time = max(post_validation_time, pv_time)
        for it in ["RECV_", "SEND_", ""]:
            if stage == 0 and it == "SEND_":
                continue
            if stage == config.n_stages - 1 and it == "RECV_":
                continue
            local_order[stage].append(ScheduledNode(
                type=it + "POST_VALIDATION",
                chunk=0,  # Only one chunk even for ZBV
                stage=stage,
                microbatch=0,
                start_time=post_validation_time,
                completion_time=post_validation_time,
            ))
            comm_set.comm_id[local_order[stage][-1]] = comm_set.comm_id_counter
            comm_set.comm_id_counter += 1
    return local_order


def add_communication_nodes(
        config: GraphConfig,
        comm_set: CommSet,
        local_order: List[List[ScheduledNode]]) -> List[List[ScheduledNode]]:
    assert len(local_order) == config.n_stages
    for stage in range(config.n_stages):
        comm_nodes = []
        for node in local_order[stage]:
            assert stage == node.stage
            cat = TYPE_TO_CAT.get(node.type)
            if cat not in (0, 1, 3):  # no communication for W
                continue
            cat_str = "FORWARD" if cat == 0 else "BACKWARD"

            comm_nodes.append([])
            stage_comm_nodes = comm_nodes[-1]
            def communicate(send_recv, stage_, comm_direction):
                # noinspection PyTypeChecker
                stage_comm_nodes.append(ScheduledNode(
                    type=send_recv + cat_str,
                    chunk=node.chunk,
                    stage=stage_,
                    microbatch=node.microbatch,
                    start_time=node.completion_time,
                    completion_time=node.completion_time,  # TODO: consider comm cost in completion time
                    comm_direction=comm_direction,
                ))

            if node.send_peer_stage is None:
                pass
            elif node.send_peer_stage < node.stage:
                assert node.send_peer_stage + 1 == node.stage
                communicate("SEND_", stage, CommDirection.PREV)
                communicate("RECV_", stage - 1, CommDirection.NEXT)
            else:
                assert node.send_peer_stage == node.stage + 1
                communicate("SEND_", stage, CommDirection.NEXT)
                communicate("RECV_", stage + 1, CommDirection.PREV)

        for stage_comm_nodes in comm_nodes:
            for comm_node in stage_comm_nodes:
                local_order[comm_node.stage].append(comm_node)
                comm_set.comm_id[local_order[comm_node.stage][-1]] = comm_set.comm_id_counter
            if len(stage_comm_nodes) > 0:
                comm_set.comm_id_counter += 1
    assert len(local_order) == config.n_stages
    return local_order


def reorder_communication(
        config: GraphConfig,
        comm_set: CommSet,
        local_order: List[List[ScheduledNode]]) -> List[List[ScheduledNode]]:
    assert len(local_order) == config.n_stages, f"unexpected num stages {len(local_order)}"
    for stage in range(config.n_stages):
        # For nodes with the same timestamp on the same stage, communication will be prioritized.
        def even_breaker(x: ScheduledNode):
            # Compute nodes are always delayed.
            if x.type in ['F', 'B', 'W', 'BW']:
                return comm_set.comm_id_counter
            # For comm nodes, order by their unique comm id
            return comm_set.comm_id[x]

        local_order[stage] = list(sorted(
            local_order[stage], key=lambda x: (x.start_time, even_breaker(x))
        ))
        # If a recv with intersects with previous computation, reorder them so that recv
        # is executed before computation and hence can be overlapped.
        for i in range(len(local_order[stage])):
            if i > 0 and local_order[stage][i - 1].type in {'F', 'B', 'W', 'BW'} and \
                local_order[stage][i].type.startswith('RECV') and \
                "POST_VALIDATION" not in local_order[stage][i].type and \
                local_order[stage][i].start_time <= local_order[stage][i - 1].completion_time:
                (local_order[stage][i], local_order[stage][i - 1]) = (local_order[stage][i - 1], local_order[stage][i])
        # print([(x.type, x.start_time, x.completion_time) for x in local_order[stage]])
    return local_order


def tag_rollback_communication(
        config: GraphConfig,
        local_order: List[List[ScheduledNode]]) -> List[List[ScheduledNode]]:
    local_order_with_rollback = [[] for _ in range(config.n_stages)]
    for rank in range(config.n_stages):
        rollback_comm = set()
        if rank > 0:
            for node in local_order[rank - 1]:
                if node.type == "POST_VALIDATION":
                    break
                if node.type == "SEND_FORWARD":
                    rollback_comm.add(node.microbatch)
        for node in local_order[rank]:
            # The second chunk should go after the post validation op.
            need_rollback = node.chunk == 0
            if node.type == "RECV_FORWARD" and node.microbatch in rollback_comm and need_rollback:
                rollback = True
                rollback_comm.remove(node.microbatch)
            else:
                rollback = False
            local_order_with_rollback[rank].append(ScheduledNode(
                type=node.type,
                stage=node.stage,
                microbatch=node.microbatch,
                start_time=node.start_time,
                completion_time=node.completion_time,
                chunk=node.chunk,
                comm_direction=node.comm_direction,
                rollback=rollback,
            ))
        assert len(rollback_comm) == 0
        # for node in local_order_with_rollback[rank]:
        #     print(f"{node.type}-{node.minibatch}-{int(node.rollback)}", end=', ')
        # print()
    return local_order_with_rollback


def comm_goes_down(stage, n_stages):
    recv_peer_stage = last_stage(stage)
    send_peer_stage = next_stage(stage, n_stages)
    return recv_peer_stage, send_peer_stage


def comm_goes_up(stage, n_stages):
    recv_peer_stage = next_stage(stage, n_stages)
    send_peer_stage = last_stage(stage)
    return recv_peer_stage, send_peer_stage


def last_stage(stage):
    return stage - 1 if stage > 0 else None


def next_stage(stage, n_stages):
    return stage + 1 if stage < n_stages - 1 else None
