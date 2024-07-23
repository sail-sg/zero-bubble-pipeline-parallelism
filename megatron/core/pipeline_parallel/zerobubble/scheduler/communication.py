from typing import List

from megatron.core.pipeline_parallel.zerobubble.scheduler import ScheduledNode
from megatron.core.pipeline_parallel.zerobubble.scheduler.graph import PPGraph
from megatron.core.pipeline_parallel.zerobubble.scheduler.zb import ZBGraph
from megatron.core.pipeline_parallel.zerobubble.scheduler.zbv import ZBVGraphBase


class CommSet:
    def __init__(self):
        self.comm_id = {}
        self.comm_id_counter = 0


TYPE_TO_CAT = {
    "F": 0,
    "B": 1,
    "W": 2,
}


def run_schedule_passes(
        graph: PPGraph,
        local_order: List[List[ScheduledNode]]) -> List[List[ScheduledNode]]:
    comm_set = CommSet()
    local_order = add_post_validation_nodes(graph, comm_set, local_order)
    local_order = add_communication_nodes(graph, comm_set, local_order)
    local_order = reorder_communication(graph, comm_set, local_order)
    local_order = tag_rollback_communication(graph, local_order)
    return local_order


def add_post_validation_nodes(
        graph: PPGraph,
        comm_set: CommSet,
        local_order: List[List[ScheduledNode]]) -> List[List[ScheduledNode]]:
    assert len(local_order) == graph.n_stages

    post_validation_time = 0
    for stage_ in range(graph.n_stages - 1, -1, -1):
        pv_time = graph.get_post_validation_time(stage_, local_order)
        post_validation_time = max(post_validation_time, pv_time)
        for it in ["RECV_", "SEND_", ""]:
            if stage_ == 0 and it == "SEND_":
                continue
            if stage_ == graph.n_stages - 1 and it == "RECV_":
                continue
            # stage_ = i - 1 if it == "RECV_" else i
            local_order[stage_].append(ScheduledNode(
                type=it + "POST_VALIDATION",
                chunk=0,  # Only one chunk even for ZBV
                stage=stage_,
                minibatch=0,
                start_time=post_validation_time,
                completion_time=post_validation_time,
            ))
            comm_set.comm_id[local_order[stage_][-1]] = comm_set.comm_id_counter
            comm_set.comm_id_counter += 1
    return local_order


def add_communication_nodes(
        graph: PPGraph,
        comm_set: CommSet,
        local_order: List[List[ScheduledNode]]) -> List[List[ScheduledNode]]:
    assert len(local_order) == graph.n_stages
    for stage in range(graph.n_stages):
        for node in local_order[stage]:
            cat = TYPE_TO_CAT.get(node.type)
            if cat not in (0, 1):  # no communication for W
                continue
            cat_str = "FORWARD" if cat == 0 else "BACKWARD"

            def communicate(send_recv, stage_):
                # noinspection PyTypeChecker
                local_order[stage_].append(ScheduledNode(
                    type=send_recv + cat_str,
                    chunk=node.chunk,
                    stage=stage_,
                    minibatch=node.minibatch,
                    start_time=node.completion_time,
                    completion_time=node.completion_time,  # TODO: consider comm cost in completion time
                ))
                comm_set.comm_id[local_order[stage_][-1]] = comm_set.comm_id_counter

            if isinstance(graph, ZBVGraphBase):
                chunk_index = node.chunk if cat == 0 else graph.config.max_chunks - 1 - node.chunk
                if chunk_index % 2 == 1 and stage > 0:
                    communicate("SEND_", stage)
                    communicate("RECV_", stage - 1)
                if chunk_index % 2 == 0 and stage < graph.n_stages - 1:
                    communicate("SEND_", stage)
                    communicate("RECV_", stage + 1)
            elif isinstance(graph, ZBGraph):
                if node.type == 'F' and node.stage != graph.n_stages - 1:
                    communicate("SEND_", stage)
                    communicate("RECV_", stage + 1)
                elif node.type == 'B' and node.stage != 0:
                    communicate("SEND_", stage)
                    communicate("RECV_", stage - 1)
            else:
                raise TypeError(f"unsupported graph type {type(graph)}")
            comm_set.comm_id_counter += 1
    return local_order


def reorder_communication(
        graph: PPGraph,
        comm_set: CommSet,
        local_order: List[List[ScheduledNode]]) -> List[List[ScheduledNode]]:
    assert len(local_order) == graph.n_stages
    for stage in range(graph.n_stages):
        # For nodes with the same timestamp on the same stage, communication will be prioritized.
        def even_breaker(x: ScheduledNode):
            # Compute nodes are always delayed.
            if x.type in ['F', 'B', 'W']:
                return comm_set.comm_id_counter
            # For comm nodes, order by their unique comm id
            return comm_set.comm_id[x]

        local_order[stage] = list(sorted(
            local_order[stage], key=lambda x: (x.start_time, even_breaker(x))
        ))
        # If a recv with intersects with previous computation, reorder them so that recv
        # is executed before computation and hence can be overlapped.
        for i in range(len(local_order[stage])):
            if i > 0 and local_order[stage][i - 1].type in {'F', 'B', 'W'} and \
                local_order[stage][i].type.startswith('RECV') and \
                "POST_VALIDATION" not in local_order[stage][i].type and \
                local_order[stage][i].start_time <= local_order[stage][i - 1].completion_time:
                (local_order[stage][i], local_order[stage][i - 1]) = (local_order[stage][i - 1], local_order[stage][i])
        # print([(x.type, x.start_time, x.completion_time) for x in local_order[stage]])
    return local_order


def tag_rollback_communication(
        graph: PPGraph,
        local_order: List[List[ScheduledNode]]) -> List[List[ScheduledNode]]:
    local_order_with_rollback = [[] for _ in range(graph.n_stages)]
    for rank in range(graph.n_stages):
        rollback_comm = set()
        if rank > 0:
            for node in local_order[rank - 1]:
                if node.type == "POST_VALIDATION":
                    break
                if node.type == "SEND_FORWARD":
                    rollback_comm.add(node.minibatch)
        for node in local_order[rank]:
            need_rollback = isinstance(graph, ZBVGraphBase) and node.chunk == 0 \
                            or isinstance(graph, ZBGraph)
            if node.type == "RECV_FORWARD" and node.minibatch in rollback_comm and need_rollback:
                rollback = True
                rollback_comm.remove(node.minibatch)
            else:
                rollback = False
            local_order_with_rollback[rank].append(ScheduledNode(
                type=node.type,
                chunk=node.chunk,
                stage=node.stage,
                minibatch=node.minibatch,
                start_time=node.start_time,
                completion_time=node.completion_time,
                rollback=rollback,
            ))
        if isinstance(graph, ZBGraph):
            assert len(rollback_comm) == 0
        # for node in local_order_with_rollback[rank]:
        #     print(f"{node.type}-{node.minibatch}-{int(node.rollback)}", end=', ')
        # print()
    return local_order_with_rollback

