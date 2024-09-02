import dataclasses
from typing import List

from megatron.core.pipeline_parallel.zerobubble.scheduler.graph import GraphConfig, F, B, BW, FuncType, ScheduledNode, CommDirection, NodeKey


class CommSet:
    def __init__(self):
        self.comm_id = {}
        self.comm_id_counter = 0


def run_communication_passes(
        config: GraphConfig,
        local_order: List[List[ScheduledNode]]) -> List[List[ScheduledNode]]:
    comm_set = CommSet()
    local_order = add_post_validation_nodes(config, comm_set, local_order)
    local_order = add_communication_nodes(config, comm_set, local_order)
    local_order = reorder_communication(config, comm_set, local_order)
    local_order = tag_rollback_communication(config, local_order)
    return local_order


def get_post_validation_time(config: GraphConfig, stage, local_order):
    node_map = {n.get_key(): n for n in sum(local_order, [])}
    deadline_idx = next(i for i, n in enumerate(local_order[stage]) if n.type != F or n.chunk != 0)
    pv_id = min(2 * (config.n_stages - 1 - stage), config.n_micro - 1)
    pv_id = min(pv_id, deadline_idx - 1)
    end_time = node_map[NodeKey(F, chunk=0, stage=stage, minibatch=pv_id)].completion_time
    func_type = local_order[stage][pv_id].type
    cost = config.get_cost(stage, func_type)
    return end_time - cost - config.cost_comm


def add_post_validation_nodes(
        config: GraphConfig,
        comm_set: CommSet,
        local_order: List[List[ScheduledNode]]) -> List[List[ScheduledNode]]:
    assert len(local_order) == config.n_stages

    pv_types = [
        FuncType.RECV_POST_VALIDATION,
        FuncType.SEND_POST_VALIDATION,
        FuncType.POST_VALIDATION,
    ]
    post_validation_time = 0
    for stage in range(config.n_stages - 1, -1, -1):
        pv_time = get_post_validation_time(config, stage, local_order)
        post_validation_time = max(post_validation_time, pv_time)
        for it in pv_types:
            if stage == 0 and it == FuncType.SEND_POST_VALIDATION:
                continue
            if stage == config.n_stages - 1 and it == FuncType.RECV_POST_VALIDATION:
                continue
            local_order[stage].append(ScheduledNode(
                type=it,
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
            if node.type not in (F, B, BW):  # no communication for W
                continue
            cat_str = "FORWARD" if node.type == F else "BACKWARD"

            comm_nodes.append([])
            stage_comm_nodes = comm_nodes[-1]
            def communicate(send_recv, stage_, comm_direction):
                # noinspection PyTypeChecker
                stage_comm_nodes.append(ScheduledNode(
                    type=FuncType(send_recv + cat_str),
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
            if x.type.is_computation():
                return comm_set.comm_id_counter
            # For comm nodes, order by their unique comm id
            return comm_set.comm_id[x]

        local_order[stage] = list(sorted(
            local_order[stage], key=lambda x: (x.start_time, even_breaker(x))
        ))
        # If a recv with intersects with previous computation, reorder them so that recv
        # is executed before computation and hence can be overlapped.
        for i in range(len(local_order[stage])):
            if i > 0 and local_order[stage][i - 1].type.is_computation() and \
                    local_order[stage][i].type.is_recv() and \
                    not local_order[stage][i].type.is_post_validation_related() and \
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
                if node.type == FuncType.POST_VALIDATION:
                    break
                if node.type == FuncType.SEND_FORWARD:
                    rollback_comm.add(node.microbatch)
        for node in local_order[rank]:
            # The second chunk should go after the post validation op.
            need_rollback = node.chunk == 0
            if node.type == FuncType.RECV_FORWARD and node.microbatch in rollback_comm and need_rollback:
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
