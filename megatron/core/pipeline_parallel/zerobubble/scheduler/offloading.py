import dataclasses
from typing import List, Tuple, Optional

from megatron.core.pipeline_parallel.zerobubble.scheduler.graph import GraphConfig, F, B, W, BW, R, ScheduledNode, FuncType


def get_offload_key(node: ScheduledNode):
    return node.layer_group_idx, node.microbatch


def get_send_recv_queue(stage_nodes: List[ScheduledNode], d2h_time: float, h2d_time: float):
    invalid_offload_keys = set()
    send_node_map = {}
    for node in stage_nodes:
        if node.type == F and node.should_offload:
            send_node_map[get_offload_key(node)] = node
    for node in stage_nodes:
        if node.type == B and node.should_offload:
            key = get_offload_key(node)
            assert key in send_node_map
            send_st = send_node_map[key].completion_time
            if node.start_time - h2d_time <= send_st + d2h_time:
                invalid_offload_keys.add(key)

    send_index_map = {}
    send_queue = []
    cur_time = 0
    for node in stage_nodes:
        if node.type == F and node.should_offload:
            if get_offload_key(node) in invalid_offload_keys:
                continue
            cur_time = max(cur_time, node.completion_time)
            send_index_map[get_offload_key(node)] = len(send_queue)
            send_queue.append((node, cur_time))
            cur_time += d2h_time

    recv_queue = []
    cur_time = stage_nodes[-1].completion_time
    for node in reversed(stage_nodes):
        if node.type == B and node.should_offload:
            if get_offload_key(node) in invalid_offload_keys:
                continue
            cur_time = min(cur_time, node.start_time)
            _, send_st = send_queue[send_index_map[get_offload_key(node)]]
            assert cur_time - h2d_time > send_st + d2h_time
            cur_time -= h2d_time
            recv_queue.append((node, cur_time))
    recv_queue = list(reversed(recv_queue))

    send_recv_queue = []
    cur_time = 0
    si, ri = 0, 0
    while si < len(send_queue):
        s_node, s_start_time = send_queue[si]
        if s_start_time > cur_time:
            last_time = s_start_time
            for i in range(len(send_recv_queue) - 1, -1, -1):
                node_i, st_time_i = send_recv_queue[i]
                if node_i.type == F:
                    break
                last_time = min(last_time, node_i.start_time)
                last_time -= h2d_time
                assert st_time_i <= last_time
                send_recv_queue[i] = node_i, last_time
        cur_time = max(cur_time, s_start_time)
        send_recv_queue.append((s_node, cur_time))
        cur_time += d2h_time
        si += 1
        if si >= len(send_queue):
            break
        next_s_node, next_s_start_time = send_queue[si]
        next_s_start_time = max(next_s_start_time, cur_time)
        while True:
            check_fail = False
            check_start_time = next_s_start_time + d2h_time
            sj = si + 1
            for rj in range(ri, len(recv_queue)):
                rj_node, rj_start_time = recv_queue[rj]
                its_key = get_offload_key(rj_node)
                its_send_index = send_index_map[its_key]
                while sj <= its_send_index:
                    sj_node, sj_start_time = send_queue[sj]
                    check_start_time = max(check_start_time, sj_start_time)
                    check_start_time += d2h_time
                    sj += 1
                if check_start_time > rj_start_time:
                    check_fail = True
                    break
                check_start_time += h2d_time
            if check_fail:
                r_node, r_start_time = recv_queue[ri]
                assert r_start_time >= cur_time
                assert send_index_map[get_offload_key(r_node)] < si
                send_recv_queue.append((r_node, cur_time))
                cur_time += h2d_time
                next_s_start_time = max(next_s_start_time, cur_time)
                ri += 1
            else:
                break
    while ri < len(recv_queue):
        r_node, r_start_time = recv_queue[ri]
        assert r_start_time >= cur_time
        assert send_index_map[get_offload_key(r_node)] < si
        send_recv_queue.append((r_node, r_start_time))
        cur_time = r_start_time + h2d_time
        ri += 1

    send_recv_queue_with_end_time = []
    for node, st_time in send_recv_queue:
        if node.type == F:
            end_time = st_time + d2h_time
        else:
            end_time = st_time + h2d_time
        send_recv_queue_with_end_time.append((node, st_time, end_time))

    return send_recv_queue_with_end_time


def add_send_recv_in_schedule(stage_nodes: List[ScheduledNode], send_recv_queue: List[Tuple[ScheduledNode, int, int]]):
    new_schedule = []
    prev_send_node: Optional[ScheduledNode] = None
    prev_end_time = 0
    recv_node_map = {}
    f_node_map = {}
    offload_idx = 0
    node_idx = 0
    while node_idx < len(stage_nodes):
        node = stage_nodes[node_idx]
        while offload_idx < len(send_recv_queue):
            offload_node, st_time, end_time = send_recv_queue[offload_idx]
            if st_time > node.start_time:
                break
            if prev_send_node is not None:
                new_schedule.append(dataclasses.replace(
                    prev_send_node,
                    type=FuncType.OFFLOAD_SEND_END,
                    start_time=prev_end_time,
                    completion_time=prev_end_time,
                ))
                prev_send_node = None
            if offload_node.type == F:
                f_node, f_end_time = f_node_map[get_offload_key(offload_node)]
                assert st_time >= f_end_time
                new_schedule.append(dataclasses.replace(
                    offload_node,
                    type=FuncType.OFFLOAD_SEND_START,
                    start_time=st_time,
                    completion_time=st_time,
                ))
                prev_send_node = offload_node
                prev_end_time = end_time
            else:
                new_schedule.append(dataclasses.replace(
                    offload_node,
                    type=FuncType.OFFLOAD_RECV_START,
                    start_time=st_time,
                    completion_time=st_time,
                ))
                recv_node_map[get_offload_key(offload_node)] = offload_node, end_time
            offload_idx += 1
        if prev_send_node is not None and prev_end_time <= node.start_time:
            new_schedule.append(dataclasses.replace(
                prev_send_node,
                type=FuncType.OFFLOAD_SEND_END,
                start_time=prev_end_time,
                completion_time=prev_end_time,
            ))
            prev_send_node = None

        new_schedule.append(node)
        node_idx += 1
        while offload_idx < len(send_recv_queue):
            offload_node, st_time, end_time = send_recv_queue[offload_idx]
            if st_time >= node.completion_time:
                break
            if prev_send_node is not None:
                new_schedule.append(dataclasses.replace(
                    prev_send_node,
                    type=FuncType.OFFLOAD_SEND_END,
                    start_time=prev_end_time,
                    completion_time=prev_end_time,
                ))
                prev_send_node = None
            if offload_node.type == F:
                f_node, f_end_time = f_node_map[get_offload_key(offload_node)]
                assert st_time >= f_end_time
                new_schedule.append(dataclasses.replace(
                    offload_node,
                    type=FuncType.OFFLOAD_SEND_START,
                    start_time=st_time,
                    completion_time=st_time,
                ))
                prev_send_node = offload_node
                prev_end_time = end_time
            else:
                new_schedule.append(dataclasses.replace(
                    offload_node,
                    type=FuncType.OFFLOAD_RECV_START,
                    start_time=st_time,
                    completion_time=st_time,
                ))
                recv_node_map[get_offload_key(offload_node)] = offload_node, end_time
            offload_idx += 1
        if prev_send_node is not None and prev_end_time <= node.completion_time:
            new_schedule.append(dataclasses.replace(
                prev_send_node,
                type=FuncType.OFFLOAD_SEND_END,
                start_time=prev_end_time,
                completion_time=prev_end_time,
            ))
            prev_send_node = None

        if node.type == W and get_offload_key(node) in recv_node_map:
            recv_node, recv_end_time = recv_node_map[get_offload_key(node)]
            # assert recv_end_time <= node.start_time
            new_schedule.append(dataclasses.replace(
                recv_node,
                type=FuncType.OFFLOAD_RECV_END,
                start_time=node.completion_time,
                completion_time=node.completion_time,
            ))
        elif node.type == F:
            f_node_map[get_offload_key(node)] = node, node.completion_time
    return new_schedule


def remove_unnecessary_offload(stage_nodes: List[ScheduledNode]) -> List[ScheduledNode]:
    offload_keys = set()
    for node in stage_nodes:
        if node.type.is_offload():
            offload_keys.add(get_offload_key(node))
    new_schedule = []
    for node in stage_nodes:
        if not node.type.is_offload() and node.should_offload and get_offload_key(node) not in offload_keys:
            new_schedule.append(dataclasses.replace(
                node, should_offload=False
            ))
        else:
            new_schedule.append(node)
    return new_schedule


def add_offload(config: GraphConfig, local_order: List[List[ScheduledNode]]) -> List[List[ScheduledNode]]:
    from megatron.training import get_args
    if not get_args().cpu_offload:
        return local_order
    offload_time = get_args().offload_time
    new_local_order = []
    for stage in range(len(local_order)):
        d2h_time = (config.cost_f[stage] + config.cost_b[stage] + config.cost_w[stage]) * offload_time
        h2d_time = d2h_time
        send_recv_queue = get_send_recv_queue(local_order[stage], d2h_time, h2d_time)
        new_schedule = add_send_recv_in_schedule(
            local_order[stage], send_recv_queue
        )
        new_schedule = remove_unnecessary_offload(new_schedule)
        new_local_order.append(new_schedule)
    return new_local_order
