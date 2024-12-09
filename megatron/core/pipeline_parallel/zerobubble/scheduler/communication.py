import dataclasses
import functools
import math
from collections import defaultdict
from typing import List, Tuple

from megatron.core.pipeline_parallel.zerobubble.scheduler.graph import GraphConfig, F, B, BW, FuncType, ScheduledNode, CommDirection
from megatron.training import get_args


class CommSet:
    def __init__(self):
        self.comm_id = {}
        self.comm_id_counter = 0


def run_communication_passes(
    config: GraphConfig,
    local_order: List[List[ScheduledNode]],
    post_validation: bool,
) -> List[List[ScheduledNode]]:
    comm_set = CommSet()
    # TODO: Remove this once we confirm add_post_validation_nodes_before_deadline works
    if post_validation:
        local_order = add_post_validation_nodes(config, comm_set, local_order)
    local_order = add_communication_nodes(config, comm_set, local_order)
    local_order = reorder_communication(config, comm_set, local_order)
    if post_validation:
        # local_order = add_post_validation_nodes_before_deadline(config, comm_set, local_order)
        local_order = tag_rollback_communication(config, local_order)
    return local_order


def get_post_validation_time(config: GraphConfig, stage, local_order: List[List[ScheduledNode]]):
    deadline_idx = next(i for i, n in enumerate(local_order[stage]) if n.type != F or n.chunk != 0)
    pv_id = min(2 * (config.n_stages - 1 - stage), config.n_micro - 1)
    pv_id = min(pv_id, deadline_idx - 1)
    end_node = next((n for n in local_order[stage]
                     if n.type == F and n.chunk == 0 and n.microbatch == pv_id and n.seq_split_idx == 0), None)
    assert end_node, f"node of first chunk not found. stage {stage} microbatch {pv_id}"
    end_time = end_node.completion_time
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
            comm_peer_stage = None
            if it == FuncType.SEND_POST_VALIDATION:
                comm_peer_stage = stage - 1
            elif it == FuncType.RECV_POST_VALIDATION:
                comm_peer_stage = stage + 1
            local_order[stage].append(ScheduledNode(
                type=it,
                chunk=0,  # Only one chunk even for ZBV
                stage=stage,
                microbatch=0,
                seq_split_idx=0,  # No sequence split for post validation
                start_time=post_validation_time,
                completion_time=post_validation_time,
                comm_peer_stage=comm_peer_stage,
            ))
            comm_set.comm_id[local_order[stage][-1]] = comm_set.comm_id_counter
            comm_set.comm_id_counter += 1
    return local_order


@dataclasses.dataclass(eq=True, frozen=True)
class CommPair:
    send_node: ScheduledNode
    recv_node: ScheduledNode
    recv_deadline: ScheduledNode

    def __post_init__(self) -> None:
        assert self.send_node
        assert self.recv_node
        assert self.recv_deadline


def add_post_validation_nodes_before_deadline(
        config: GraphConfig,
        local_order: List[List[ScheduledNode]]) -> Tuple[List[List[ScheduledNode]], List[CommPair]]:
    assert len(local_order) == config.n_stages

    pv_types = [
        FuncType.POST_VALIDATION,
        FuncType.RECV_POST_VALIDATION,
        FuncType.SEND_POST_VALIDATION,
    ]

    comm_pairs = []
    next_stage_send = None
    post_validation_time = -math.inf
    last_deadline = None
    for stage in range(config.n_stages - 1, -1, -1):
        last_post_validation_time = post_validation_time
        ddl_idx, ddl_node = next((
            (i, n) for i, n in enumerate(local_order[stage])
                if n.type.is_computation() and (n.type != F or n.chunk != 0)))
        insert_idx, origin_node = next((
            (i, n) for i, n in enumerate(local_order[stage][:ddl_idx])
                # POST_VALIDATION needs to run after any existing SEND with the same start_time,
                # or it will produce the following schedules:
                # stage0:  PL S
                # stage1:  SPL
                # All 3 nodes have the same start_time.
                # But S could possibly pop first when inserting RECVs, which pop the PL,
                # leaving the RPL inserted after PL, which is invalid.
                #
                # There could be a gap because of communication
                # so completion_time of previous node can be less than last_post_validation_time
                # so need to use start_time of next node.
                if n.start_time > last_post_validation_time),
            (ddl_idx, ddl_node)
        )
        if insert_idx > 0:
            pv_time = local_order[stage][insert_idx - 1].completion_time
        else:
            pv_time = origin_node.start_time
        if last_deadline:
            assert pv_time < last_deadline.completion_time, \
                """completion_time of POST_VALIDATION needs to be strictly less than
                the completion_time of last deadline node, which is the start_time of RECV of current deadline node,
                or the the RECV of current deadline node could go before POST_VALIDATION
                """
        last_deadline = ddl_node
        post_validation_time = max(last_post_validation_time, pv_time)

        send_node, recv_node, post_vali_node = None, None, None
        for it in pv_types:
            if stage == 0 and it == FuncType.SEND_POST_VALIDATION:
                continue
            if stage == config.n_stages - 1 and it == FuncType.RECV_POST_VALIDATION:
                continue

            comm_peer_stage = None
            if it == FuncType.SEND_POST_VALIDATION:
                comm_peer_stage = stage - 1
            elif it == FuncType.RECV_POST_VALIDATION:
                comm_peer_stage = stage + 1
            comm_pair_id = None
            # Use negative int for post validation communication
            if it == FuncType.SEND_POST_VALIDATION:
                comm_pair_id = -stage
            elif it == FuncType.RECV_POST_VALIDATION:
                comm_pair_id = -(stage + 1)
            assert comm_pair_id is None or comm_pair_id < 0
            t = last_post_validation_time if it == FuncType.RECV_POST_VALIDATION else post_validation_time
            node = ScheduledNode(
                type=it,
                chunk=0,  # Only one chunk even for ZBV
                stage=stage,
                microbatch=0,
                # Need unique layer_group_idx to make sure
                # the keys of post_validation in each stage are different
                layer_group_idx=-stage,
                seq_split_idx=0,  # No sequence split for post validation
                start_time=t,
                completion_time=t,
                comm_peer_stage=comm_peer_stage,
                comm_pair_id=comm_pair_id,
            )
            if it == FuncType.POST_VALIDATION:
                post_vali_node = node
            elif it == FuncType.RECV_POST_VALIDATION:
                recv_node = node
            else:
                assert it == FuncType.SEND_POST_VALIDATION
                send_node = node

        deadline_node = send_node or post_vali_node
        if next_stage_send:
            comm_pairs.append(CommPair(next_stage_send, recv_node, deadline_node))
        next_stage_send = send_node
        # POST_VALIDATION goes after SEND.
        # Return the RECVs.
        local_order[stage].insert(insert_idx, post_vali_node)
        if send_node:
            local_order[stage].insert(insert_idx, send_node)

    assert len(comm_pairs) == len(local_order) - 1
    return local_order, comm_pairs


def add_communication_nodes_without_sorting(
    config: GraphConfig,
    local_order: List[List[ScheduledNode]],
    post_validation: bool,
) -> List[List[ScheduledNode]]:
    local_order, comm_pairs = insert_send_nodes(config, local_order)
    if post_validation:
        local_order, post_validation_comm_pairs = add_post_validation_nodes_before_deadline(config, local_order)
        comm_pairs.extend(post_validation_comm_pairs)
    local_order = insert_recv_nodes(config, local_order, comm_pairs)
    if post_validation:
        local_order = tag_rollback_communication(config, local_order)
    return local_order


def insert_send_nodes(
    config: GraphConfig,
    local_order: List[List[ScheduledNode]]
) -> Tuple[List[List[ScheduledNode]], List[CommPair]]:
    assert len(local_order) == config.n_stages
    node_map = {n.get_key(): n for n in sum(local_order, [])}
    comm_pair_id = 0
    new_local_order = [[n for n in stage_nodes] for stage_nodes in local_order]

    comm_pairs = []

    for stage in range(config.n_stages):
        for node in local_order[stage]:
            assert stage == node.stage, f"Invalid node stage {stage} {node}"
            if node.type not in (F, B, BW):  # no communication for W
                continue
            cat_str = "FORWARD" if node.type == F else "BACKWARD"

            def create_communicate_node(send_recv, compute_node, comm_peer_stage, t, comm_direction, comm_pair_id):
                # noinspection PyTypeChecker
                return ScheduledNode(
                    type=FuncType(send_recv + cat_str),
                    chunk=compute_node.chunk,
                    stage=compute_node.stage,
                    microbatch=node.microbatch,
                    seq_split_idx=node.seq_split_idx,
                    layer_group_idx=compute_node.layer_group_idx,
                    start_time=t,
                    completion_time=t,  # TODO: consider comm cost in completion time
                    comm_direction=comm_direction,
                    comm_peer_stage=comm_peer_stage,
                    comm_pair_id=comm_pair_id,
                )

            if node.recv_peer_stage is None or node.recv_peer_stage == node.stage:
                pass
            else:
                if node.recv_peer_stage + 1 == node.stage or (node.stage == 0 and node.recv_peer_stage == config.n_stages - 1):
                    # recv from prev
                    send_direction = CommDirection.NEXT
                    recv_direction = CommDirection.PREV
                else:
                    # recv from next
                    assert node.recv_peer_stage == node.stage + 1 or \
                           (node.recv_peer_stage == 0 and node.stage == config.n_stages - 1), \
                        f"Invalid send-recv stages {node.recv_peer_stage} {node.stage}"
                    send_direction = CommDirection.PREV
                    recv_direction = CommDirection.NEXT
                peer = node_map[node.get_prev_key(config.num_layer_groups())]
                assert peer.stage == node.recv_peer_stage
                send_node = create_communicate_node("SEND_", peer, stage, peer.completion_time, send_direction, comm_pair_id)
                recv_node = create_communicate_node("RECV_", node, peer.stage, peer.completion_time, recv_direction, comm_pair_id)
                comm_pairs.append(CommPair(send_node, recv_node, recv_deadline=node))
                comm_pair_id += 1

                send_stage_nodes = new_local_order[send_node.stage]
                send_compute_pos = next((i for i, n in enumerate(send_stage_nodes) if n.get_key() == peer.get_key()))
                insert_pos = send_compute_pos + 1
                # Some offload nodes may locate after the compute node,
                # but it's start_time is the same of the compute node,
                # which is less than the start_time of the SEND here.
                insert_pos = next((
                    i + insert_pos for i, n in enumerate(send_stage_nodes[insert_pos:])
                        if n.start_time >= send_node.start_time
                ), len(send_stage_nodes))
                send_stage_nodes.insert(insert_pos, send_node)

    return new_local_order, comm_pairs


def insert_recv_nodes(
        config: GraphConfig,
        local_order: List[List[ScheduledNode]],
        comm_pairs: List[CommPair],
    ) -> List[List[ScheduledNode]]:
    from .passes import viz_node

    send_index_map = {
        p.send_node.get_key(): local_order[p.send_node.stage].index(p.send_node) for p in comm_pairs
    }

    # Add index to make sure the order is consistent with the order in each stage.
    def cmp_pair(a: CommPair, b: CommPair):
        sa, sb = a.send_node, b.send_node
        if sa.stage == sb.stage:
            return send_index_map[sa.get_key()] - send_index_map[sb.get_key()]
        elif sa.start_time == sb.start_time:
            # Let SEND_POST_VALIDATION go first
            a_type_rank = int(sa.type != FuncType.SEND_POST_VALIDATION)
            b_type_rank = int(sb.type != FuncType.SEND_POST_VALIDATION)
            return a_type_rank - b_type_rank
        return sa.start_time - sb.start_time

    # This sorting preserves the original order if start_time are the same,
    # which preserve the same behaviour of previous time-sorting based implementation for non-zb schedules.
    comm_pairs.sort(key=functools.cmp_to_key(cmp_pair))
    start_indices = [0 for _ in local_order]
    new_local_order = local_order

    for pair in comm_pairs:
        send_node = pair.send_node
        recv_node = pair.recv_node
        recv_deadline = pair.recv_deadline

        send_stage_nodes = new_local_order[send_node.stage]
        start = start_indices[send_node.stage]
        send_pos = next((i + start for i, n in enumerate(send_stage_nodes[start:]) if n.get_key() == send_node.get_key()))
        assert send_pos >= start
        start_indices[send_node.stage] = send_pos + 1

        # Pop the communication from offload to avoid conflict
        while True:
            idx, offload_barrier = next((
                (i + start, n) for i, n in enumerate(send_stage_nodes[start:send_pos])
                    if n.type.has_offload_barrier()
            ), (None, None))
            if not offload_barrier:
                break
            start = idx + 1

            peer_stage = offload_barrier.stage ^ 1
            peer_start = start_indices[peer_stage]
            peer_node_idx = None
            peer_nodes = local_order[peer_stage][peer_start:]
            for i, n in enumerate(peer_nodes):
                if n.type.is_send():
                    break
                if n.type.has_offload_barrier():
                    peer_node_idx = i + peer_start
                    break

            peer_nodes_str = ' '.join(map(viz_node, peer_nodes))
            assert peer_node_idx, \
                f"cannot find peer offload barrier. Tried to find peer of {offload_barrier} in stage {peer_stage} nodes: {peer_nodes_str}"
            start_indices[peer_stage] = peer_node_idx + 1

        recv_stage_nodes = new_local_order[recv_node.stage]
        start = start_indices[recv_node.stage]
        # There shouldn't be timespan overlap for non-communication nodes.
        recv_pos = next(
            (i + start for i, n in enumerate(recv_stage_nodes[start:])
                if (recv_node.start_time <= n.completion_time
                    or n.get_key() == recv_deadline.get_key()  # Recv should go before its consumer node.
                    or n.type.is_send() or n.type.has_offload_barrier())  # Avoid cross dependency.
             ))
        recv_stage_nodes.insert(recv_pos, recv_node)
        start_indices[recv_node.stage] = recv_pos + 1

        stage_nodes_str = ' '.join(map(viz_node, recv_stage_nodes))
        assert any(n for n in recv_stage_nodes[recv_pos+1:] if n.get_key() == recv_deadline.get_key()), \
            f"Cannot find recv consumer node after recv: stage {recv_node.stage} recv {viz_node(recv_node)} consumer {viz_node(recv_deadline)} nodes: {stage_nodes_str}"

    assert len(new_local_order) == config.n_stages
    return new_local_order


def add_communication_nodes(
        config: GraphConfig,
        comm_set: CommSet,
        local_order: List[List[ScheduledNode]]) -> List[List[ScheduledNode]]:
    assert len(local_order) == config.n_stages
    node_map = {n.get_key(): n for n in sum(local_order, [])}
    comm_pair_id = 0
    for stage in range(config.n_stages):
        comm_nodes = []
        for node in local_order[stage]:
            assert stage == node.stage, f"Invalid node stage {stage} {node}"
            if node.type not in (F, B, BW):  # no communication for W
                continue
            cat_str = "FORWARD" if node.type == F else "BACKWARD"

            comm_nodes.append([])
            stage_comm_nodes = comm_nodes[-1]

            def communicate(send_recv, compute_node, comm_peer_stage, t, comm_direction, comm_pair_id):
                # noinspection PyTypeChecker
                stage_comm_nodes.append(ScheduledNode(
                    type=FuncType(send_recv + cat_str),
                    chunk=compute_node.chunk,
                    stage=compute_node.stage,
                    microbatch=node.microbatch,
                    seq_split_idx=node.seq_split_idx,
                    layer_group_idx=compute_node.layer_group_idx,
                    start_time=t,
                    completion_time=t,  # TODO: consider comm cost in completion time
                    comm_direction=comm_direction,
                    comm_peer_stage=comm_peer_stage,
                    comm_pair_id=comm_pair_id,
                ))

            if node.recv_peer_stage is None or node.recv_peer_stage == node.stage:
                pass
            else:
                if node.recv_peer_stage + 1 == node.stage or (node.stage == 0 and node.recv_peer_stage == config.n_stages - 1):
                    # recv from prev
                    send_direction = CommDirection.NEXT
                    recv_direction = CommDirection.PREV
                else:
                    # recv from next
                    assert node.recv_peer_stage == node.stage + 1 or \
                           (node.recv_peer_stage == 0 and node.stage == config.n_stages - 1), \
                        f"Invalid send-recv stages {node.recv_peer_stage} {node.stage}"
                    send_direction = CommDirection.PREV
                    recv_direction = CommDirection.NEXT
                peer = node_map[node.get_prev_key(config.num_layer_groups())]
                assert peer.stage == node.recv_peer_stage
                communicate("SEND_", peer, stage, peer.completion_time, send_direction, comm_pair_id)
                communicate("RECV_", node, peer.stage, peer.completion_time, recv_direction, comm_pair_id)
                comm_pair_id += 1

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
    local_order: List[List[ScheduledNode]],
) -> List[List[ScheduledNode]]:
    assert len(local_order) == config.n_stages, f"unexpected num stages {len(local_order)}"
    for stage in range(config.n_stages):
        non_comm_nodes = [n for n in local_order[stage]
                          if not n.type.is_communication() and not n.type.is_post_validation_related()]

        # For nodes with the same timestamp on the same stage, communication will be prioritized.
        def even_breaker(x: ScheduledNode):
            # Compute and Offload nodes are always delayed.
            # This requires the sorting to be stable
            # so that it won't change the order of non-communication nodes.
            if not x.type.is_communication():
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

        # The reordering must not reorder the origin non-comm nodes.
        new_non_comm_nodes = [n for n in local_order[stage]
                              if not n.type.is_communication() and not n.type.is_post_validation_related()]
        assert len(new_non_comm_nodes) == len(non_comm_nodes)
        for n, o in zip(new_non_comm_nodes, non_comm_nodes):
            assert n == o, f"{n} | {o}"
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
                assert node.type not in (FuncType.RECV_BACKWARD, FuncType.SEND_BACKWARD), \
                    f"found {node.type} before POST_VALIDATION"
                if node.type in (FuncType.RECV_FORWARD, FuncType.SEND_FORWARD):
                    assert node.chunk == 0, f"found node with chunk > 0 before POST_VALIDATION: {node}"
        for node in local_order[rank]:
            # The second chunk should go after the post validation op.
            need_rollback = node.chunk == 0
            if node.type == FuncType.RECV_FORWARD and node.microbatch in rollback_comm and need_rollback:
                rollback = True
                rollback_comm.remove(node.microbatch)
            else:
                rollback = False
            local_order_with_rollback[rank].append(dataclasses.replace(node, rollback=rollback))
        assert len(rollback_comm) == 0
        # for node in local_order_with_rollback[rank]:
        #     print(f"{node.type}-{node.minibatch}-{int(node.rollback)}", end=', ')
        # print()
    return local_order_with_rollback


def validate_communication(local_order: List[List[ScheduledNode]], debug=False):
    # Fuse kernel
    fused_comm = []
    n_comm = 0
    for nodes in local_order:
        comms = []
        curr_comm = set()
        for n in nodes:
            if n.type.is_send() or n.type.is_recv():
                assert n not in curr_comm
                curr_comm.add(n)
                continue
            if curr_comm:
                comms.append(curr_comm)
                n_comm += len(curr_comm)
                curr_comm = set()
        if curr_comm:
            comms.append(curr_comm)
            n_comm += len(curr_comm)
        fused_comm.append(comms)
    assert len(fused_comm) == len(local_order)

    stage_curr_index = [0 for _ in fused_comm]
    ct = 0
    last_found = True
    while ct < n_comm:
        found = False
        pending_comm = {}
        for stage in range(len(fused_comm)):
            if stage_curr_index[stage] >= len(fused_comm[stage]):
                continue

            # Copy it as we need to modify it inside the loop.
            curr_fused_nodes = list(fused_comm[stage][stage_curr_index[stage]])
            for node in curr_fused_nodes:
                assert node.stage == stage, f"stage: {stage} node: {node}"

                if debug and last_found:
                    print_remaining_nodes(fused_comm)

                assert node.comm_peer_stage is not None
                # Different chunk for interleaved pipeline parallel
                peer_key = node.comm_pair_id
                if peer_key not in pending_comm:
                    node_key = node.comm_pair_id
                    pending_comm[node_key] = node
                    last_found = False
                    continue

                found = True
                last_found = True
                ct += 2
                peer = pending_comm.pop(peer_key)
                for n in [node, peer]:
                    fused = fused_comm[n.stage][stage_curr_index[n.stage]]
                    fused.remove(n)
                    if not fused:
                        stage_curr_index[n.stage] += 1
        if not found:
            raise RuntimeError(f"Cannot find next runnable node. Pending: {pending_comm}")


def print_remaining_nodes(fused_comm):
    from megatron.core.pipeline_parallel.zerobubble.scheduler.passes import viz_node
    print(f"=" * 30)
    for _stage in range(len(fused_comm)):
        ss = []
        for fused in fused_comm[_stage]:
            if fused:
                ss.append(' '.join(viz_node(f) for f in fused))
        print(f"{_stage}: {','.join(ss)}")
