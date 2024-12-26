import dataclasses
from dataclasses import dataclass
from typing import List, Tuple, Optional


from megatron.core.pipeline_parallel.zerobubble.scheduler.graph import GraphConfig, F, B, W, BW, R, ScheduledNode, FuncType, NodeKey


def get_offload_key(node: ScheduledNode):
    return node.layer_group_idx, node.microbatch

def get_offload_overlap_sr():
    from megatron.training import get_args
    return get_args().offload_overlap_sr


@dataclass(eq=True)
class OffloadPass:
    index: int
    node: ScheduledNode
    start_time: int
    completion_time: int


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


def add_barriers_before_offload(stage_0, stage_1, idx):
    assert stage_0[0].type == F
    eps = 1e-6
    new_stage_0, new_stage_1 = [], []
    j_0, j_1 = 0, 0
    while True:
        i_0, i_1 = j_0, j_1
        while i_0 < len(stage_0) and stage_0[i_0].type not in [FuncType.OFFLOAD_SEND_START, FuncType.OFFLOAD_RECV_START]:
            i_0 += 1
        while i_1 < len(stage_1) and stage_1[i_1].type not in [FuncType.OFFLOAD_SEND_START, FuncType.OFFLOAD_RECV_START]:
            i_1 += 1
        if i_0 >= len(stage_0) and i_1 >= len(stage_1):
            break
        cur_time = stage_0[-1].completion_time
        if i_0 < len(stage_0):
            cur_time = min(cur_time, stage_0[i_0].start_time)
        if i_1 < len(stage_1):
            cur_time = min(cur_time, stage_1[i_1].start_time)

        while j_0 < i_0 and stage_0[j_0].start_time <= cur_time + eps:
            new_stage_0.append(stage_0[j_0])
            j_0 += 1
        if i_0 >= len(stage_0) or stage_0[i_0].start_time > cur_time + eps:
            new_stage_0.append(ScheduledNode(
                type=FuncType.OFFLOAD_BARRIER,
                stage=idx,
                microbatch=0,
                start_time=cur_time,
                completion_time=cur_time,
            ))
        if i_0 < len(stage_0) and stage_0[i_0].start_time <= cur_time + eps:
            assert i_0 == j_0
            new_stage_0.append(stage_0[j_0])
            j_0 += 1

        while j_1 < i_1 and stage_1[j_1].start_time <= cur_time + eps:
            new_stage_1.append(stage_1[j_1])
            j_1 += 1
        if i_1 >= len(stage_1) or stage_1[i_1].start_time > cur_time + eps:
            new_stage_1.append(ScheduledNode(
                type=FuncType.OFFLOAD_BARRIER,
                stage=idx + 1,
                microbatch=0,
                start_time=cur_time,
                completion_time=cur_time,
            ))
        if i_1 < len(stage_1) and stage_1[i_1].start_time <= cur_time + eps:
            assert i_1 == j_1
            new_stage_1.append(stage_1[j_1])
            j_1 += 1
    while j_0 < len(stage_0):
        new_stage_0.append(stage_0[j_0])
        j_0 += 1
    while j_1 < len(stage_1):
        new_stage_1.append(stage_1[j_1])
        j_1 += 1
    return new_stage_0, new_stage_1


def get_peak_memory(local_order: List[List[ScheduledNode]]):
    peak_memory_all_ranks = []
    for stage_nodes in local_order:
        peak, mem = 0, 0
        for node in stage_nodes:
            if node.type in [F, FuncType.OFFLOAD_RECV_PREP]:
                mem += 1
            elif node.type in [BW, W, FuncType.OFFLOAD_SEND_END]:
                mem -= 1
            peak = max(peak, mem)
        peak_memory_all_ranks.append(peak)
    return peak_memory_all_ranks


def add_send_recv(stage_nodes: List[ScheduledNode], starting_time: int, offload_time: int):
    h2d_time = d2h_time = offload_time

    # remove invalid pairs
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
            if node.start_time - (node.completion_time - node.start_time) - send_st <= (h2d_time + d2h_time) * 3:
                invalid_offload_keys.add(key)

    send_queue = []
    cur_time = starting_time
    cur_index = 0
    send_index_map = {}
    for node in stage_nodes:
        if node.type == F and node.should_offload:
            if get_offload_key(node) in invalid_offload_keys:
                continue
            while cur_time < node.completion_time:
                cur_time += h2d_time + d2h_time
                cur_index += 2
            send_index_map[get_offload_key(node)] = len(send_queue)
            send_queue.append(OffloadPass(cur_index, node, cur_time, cur_time + d2h_time))
            cur_time += h2d_time + d2h_time
            cur_index += 2

    while cur_time < stage_nodes[-1].completion_time:
        cur_time += h2d_time + d2h_time
        cur_index += 2

    recv_queue = []
    for node in reversed(stage_nodes):
        if node.type == B and node.should_offload:
            if get_offload_key(node) in invalid_offload_keys:
                continue
            while cur_time > node.start_time - (node.completion_time - node.start_time): # buffer for robustness
                cur_time -= h2d_time + d2h_time
                cur_index -= 2
            send_st = send_queue[send_index_map[get_offload_key(node)]].start_time
            assert cur_time - h2d_time > send_st + d2h_time
            recv_queue.append(OffloadPass(cur_index - 1, node, cur_time - h2d_time, cur_time))
            cur_time -= h2d_time + d2h_time
            cur_index -= 2
    recv_queue = list(reversed(recv_queue))

    send_recv_queue = sorted(send_queue + recv_queue, key=lambda x: x.index)
    return send_recv_queue

    def get_send_index(_time):
        # min send index with start_time >= _time
        _index = (_time - starting_time) // (h2d_time + d2h_time) * 2
        if _index // 2 * (h2d_time + d2h_time) < _time - starting_time:
            _index += 2
        return _index

    def get_recv_index(_time):
        # max recv index with start_time <= _time
        return (_time - starting_time - d2h_time) // (h2d_time + d2h_time) * 2 + 1

    def get_start_time(_index):
        _time = starting_time + _index // 2 * (h2d_time + d2h_time)
        if _index % 2 == 1:
            _time += d2h_time
        return _time

    backward_node, forward_node = {}, {}
    min_backward_time = stage_nodes[-1].completion_time
    for node in stage_nodes:
        if node.type in [B, BW]:
            min_backward_time = min(min_backward_time, node.start_time)
            backward_node[get_offload_key(node)] = node
        if node.type == F:
            forward_node[get_offload_key(node)] = node

    send_map, recv_map = {}, {}
    cur_index = get_recv_index(stage_nodes[-1].completion_time)
    for node in reversed(stage_nodes):
        if node.type in [B, BW] and node.should_offload:
            _recv_index = get_recv_index(node.start_time - (node.completion_time - node.start_time)) # buffer for robustness
            cur_index = min(cur_index, _recv_index)
            recv_start = get_start_time(cur_index)
            if recv_start <= min_backward_time:
                continue
            send_node = forward_node[get_offload_key(node)]
            send_index = get_send_index(send_node.completion_time)
            if send_index + 3 > cur_index:
                continue
            recv_map[get_offload_key(node)] = OffloadPass(cur_index, node, recv_start, recv_start + h2d_time)
            cur_index -= 2
    cur_index = 0
    for node in stage_nodes:
        if node.type == F and get_offload_key(node) in recv_map:
            _send_index = get_send_index(node.completion_time)
            cur_index = max(cur_index, _send_index)
            recv_pass = recv_map[get_offload_key(node)]
            assert cur_index + 3 <= recv_pass.index  # TODO: can just delete this offload
            send_start = get_start_time(cur_index)
            send_map[get_offload_key(node)] = OffloadPass(cur_index, node, send_start, send_start + d2h_time)
            cur_index += 2

    send_recv_queue = sorted(list(send_map.values()) + list(recv_map.values()), key=lambda x: x.index)
    return send_recv_queue


def adjust_warmup(send_recv_0: List[OffloadPass], send_recv_1: List[OffloadPass], solo_d2h: int):
    min_recv_index = send_recv_0[-1].index
    max_send_index = -1

    for rank, send_recv in enumerate([send_recv_0, send_recv_1]):
        for _pass in send_recv:
            if _pass.index % 2 == 1:
                min_recv_index = min(min_recv_index, _pass.index + rank)
            else:
                max_send_index = max(max_send_index, _pass.index + rank)

    # adjust warmup
    cur_time = send_recv_0[0].start_time
    cur_index = send_recv_0[0].index
    sr_index = [0, 0]
    while cur_index < min_recv_index:
        found = False
        for rank, send_recv in enumerate([send_recv_0, send_recv_1]):
            if send_recv[sr_index[rank]].index + rank == cur_index:
                assert not found
                found = True
                send_node = send_recv[sr_index[rank]]
                cur_time = max(cur_time, send_node.node.completion_time)
                assert cur_time <= send_node.start_time
                send_recv[sr_index[rank]] = dataclasses.replace(
                    send_node, start_time=cur_time, completion_time=cur_time + solo_d2h)
                sr_index[rank] += 1
                cur_time += solo_d2h
        cur_index += 1

    return send_recv_0, send_recv_1


def add_send_recv_in_schedule(stage_nodes: List[ScheduledNode], send_recv_queue: List[OffloadPass]):
    left_queue = []
    right_queue = []
    send_index_map = {}
    for sr_pass in send_recv_queue:
        node = sr_pass.node
        st_time = sr_pass.start_time
        if node.type == F:
            send_index_map[get_offload_key(node)] = sr_pass
            left_queue.append(dataclasses.replace(
                node,
                type=FuncType.OFFLOAD_SEND_START,
                start_time=st_time,
                completion_time=st_time
            ))
            right_queue.append(dataclasses.replace(
                node,
                type=FuncType.OFFLOAD_SEND_END,
                start_time=sr_pass.completion_time,
                completion_time=sr_pass.completion_time
            ))
        else:
            left_queue.append(dataclasses.replace(
                node,
                type=FuncType.OFFLOAD_RECV_PREP,
                start_time=st_time,
                completion_time=st_time,
            ))
            right_queue.append(dataclasses.replace(
                node,
                type=FuncType.OFFLOAD_RECV_START,
                start_time=st_time,
                completion_time=st_time,
            ))
    new_schedule = []
    l_idx, r_idx = 0, 0
    new_nodes = []
    priority = {
        FuncType.OFFLOAD_SEND_START: 0,
        FuncType.OFFLOAD_SEND_END: 1,
        FuncType.OFFLOAD_RECV_PREP: 2,
        FuncType.OFFLOAD_RECV_START: 3,
        FuncType.OFFLOAD_RECV_END: 4,
    }
    for node in stage_nodes:
        while r_idx < len(right_queue):
            right_node = right_queue[r_idx]
            if right_node.completion_time > node.start_time:
                break
            new_nodes.append(right_node)
            r_idx += 1
        while l_idx < len(left_queue):
            left_node = left_queue[l_idx]
            if left_node.start_time >= node.completion_time:
                break
            new_nodes.append(left_node)
            l_idx += 1
        new_nodes = sorted(new_nodes, key=lambda x: (x.start_time, priority[x.type]))
        new_schedule += new_nodes
        new_nodes = []
        new_schedule.append(node)
        if node.type in [W, BW] and get_offload_key(node) in send_index_map:
            new_nodes.append(dataclasses.replace(
                node,
                type=FuncType.OFFLOAD_RECV_END,
                start_time=node.completion_time,
                completion_time=node.completion_time,
            ))
    new_schedule += new_nodes
    assert l_idx >= len(left_queue) and r_idx >= len(right_queue)
    return new_schedule


def add_offload_passes(stage_0: List[ScheduledNode], stage_1: List[ScheduledNode], rank: int, parallel_offload_time: int, solo_offload_time: int):
    assert stage_0[0].type == F
    start_time = stage_0[0].completion_time

    send_recv_0 = add_send_recv(stage_0, start_time, parallel_offload_time)
    send_recv_1 = add_send_recv(stage_1, start_time + parallel_offload_time, parallel_offload_time)

    send_recv_0, send_recv_1 = adjust_warmup(send_recv_0, send_recv_1, solo_offload_time)

    stage_0 = add_send_recv_in_schedule(stage_0, send_recv_0)
    stage_0 = remove_unnecessary_offload(stage_0)

    stage_1 = add_send_recv_in_schedule(stage_1, send_recv_1)
    stage_1 = remove_unnecessary_offload(stage_1)

    stage_0, stage_1 = add_barriers_before_offload(stage_0, stage_1, rank)
    return stage_0, stage_1


def postpone_forward_in_warmup(config: GraphConfig, local_order: List[List[ScheduledNode]]):
    node_map = {}
    rank_index = [-1] * len(local_order)
    max_chunk = 0
    for rank, stage_nodes in enumerate(local_order):
        for i, node in enumerate(stage_nodes):
            node_map[node.get_key()] = node
            max_chunk = max(max_chunk, node.chunk)
            if node.type in [B, BW] and rank_index[rank] < 0:
                rank_index[rank] = i
    for rank in range(len(local_order) - 1, -1, -1):
        first_backward_node = local_order[rank][rank_index[rank]]
        backward_cost = first_backward_node.completion_time - first_backward_node.start_time
        cur_time = first_backward_node.start_time - backward_cost  # buffer for robustness
        for i in range(rank_index[rank] - 1, -1, -1):
            node = local_order[rank][i]
            if node.type == F and node.chunk == max_chunk and node.microbatch == 0:
                break
            next_key = NodeKey(node.type, node.layer_group_idx + 1, node.microbatch, node.seq_split_idx)
            assert next_key in node_map, f"{next_key}, {node.get_key()}, {rank}"
            next_node = node_map[next_key]
            cost = node.completion_time - node.start_time
            start_time = min(next_node.start_time - config.cost_comm - cost, cur_time - cost)
            if start_time > node.start_time:
                print(rank, node.microbatch, node.chunk, node.start_time, node.completion_time, start_time, start_time + cost)
                node = dataclasses.replace(
                    node, start_time=start_time, completion_time=start_time + cost
                )
                local_order[rank][i] = node_map[node.get_key()] = node
            cur_time = node.start_time
    return local_order


def add_offload(config: GraphConfig, local_order: List[List[ScheduledNode]], offload_time) -> List[List[ScheduledNode]]:
    assert offload_time is not None
    assert len(local_order) % 2 == 0

    local_order = postpone_forward_in_warmup(config, local_order)

    parallel_offload_time = max([ft + bt + wt for ft, bt, wt in zip(config.cost_f, config.cost_b, config.cost_w)]) * offload_time
    parallel_offload_time = round(parallel_offload_time, 2)
    solo_offload_time = parallel_offload_time * 0.7
    peak_memory_before = get_peak_memory(local_order)

    new_local_order = []
    for rank in range(0, len(local_order), 2):
        stage_0 = local_order[rank]
        stage_1 = local_order[rank + 1]
        new_stage_0, new_stage_1 = add_offload_passes(stage_0, stage_1, rank, parallel_offload_time, solo_offload_time)
        new_local_order.append(new_stage_0)
        new_local_order.append(new_stage_1)

    peak_memory_all_ranks = get_peak_memory(new_local_order)
    print(f"peak memory: {peak_memory_before} -> {peak_memory_all_ranks}")
    print(f"maximum peak memory: {max(peak_memory_before)} -> {max(peak_memory_all_ranks)}")
    return new_local_order
