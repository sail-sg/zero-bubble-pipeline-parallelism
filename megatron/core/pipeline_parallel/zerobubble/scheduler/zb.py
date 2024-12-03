from dataclasses import dataclass
from typing import List, Set

import torch
import pulp

from megatron.core.pipeline_parallel.zerobubble.scheduler.graph import GraphConfig, ScheduledNode, FuncType


@dataclass
class Graph:
    nstages: int
    nmb: int
    nnodes: int
    config: GraphConfig
    parents: List[Set[int]] = None
    name: List[str] = None
    precede: torch.Tensor = None

    # ID mapping:
    # F[stage][minibatch]: 0..STAGE* MB
    # B[stage][minibatch]: STAGE* MB .. 2 * STAGE * MB
    # W[stage][minibatch]: 2 * STAGE* MB .. 3 * STAGE * MB

    def get_id(self, type, stage, mb):
        return type * (self.nstages * self.nmb) + stage * self.nmb + mb

    def get_stage(self, id):
        return (id // self.nmb) % self.nstages

    def get_cost(self, id):
        type = id // (self.nstages * self.nmb)
        stage = self.get_stage(id)
        return [self.config.cost_f[stage], self.config.cost_b[stage], self.config.cost_w[stage]][type]

    def get_mem(self, id):
        type = id // (self.nstages * self.nmb)
        stage = self.get_stage(id)
        return [self.config.mem_f[stage], self.config.mem_b[stage], self.config.mem_w[stage]][type]

    def requires_order(self, i, j):
        return (
            i != j
            and not self.precede[i][j]
            and not self.precede[j][i]
            and self.get_stage(i) == self.get_stage(j)
        )

    @classmethod
    def build_graph(cls, nstages, nmb, config):
        nnodes = nstages * nmb * 3
        g = Graph(nstages=nstages, nmb=nmb, nnodes=nnodes, config=config)
        parents = []
        name = []
        for type in range(3):
            for stage in range(nstages):
                for mb in range(nmb):
                    p = set()
                    if type == 0:
                        name.append(f'F{mb}')
                        if stage > 0:
                            p.add(g.get_id(type, stage - 1, mb))
                        if mb > 0:
                            p.add(g.get_id(type, stage, mb - 1))
                    elif type == 1:
                        name.append(f'B{mb}')
                        if stage == nstages - 1:
                            p.add(g.get_id(0, stage, mb))
                        else:
                            p.add(g.get_id(type, stage + 1, mb))
                        if mb > 0:
                            p.add(g.get_id(type, stage, mb - 1))
                    elif type == 2:
                        name.append(f'W{mb}')
                        p.add(g.get_id(1, stage, mb))
                        if mb > 0:
                            p.add(g.get_id(type, stage, mb - 1))
                    else:
                        assert False
                    parents.append(p)

        g.name = name
        g.parents = parents
        return g

    # Manual ordering producing this kind of schedule:
    # fffffffbfbfbfbfbfbwbwbwbwbwbwbwwwwww
    #  fffffbfbfbfbfbfbfbfbwbwbwbwbwwwwwwww
    #   fffbfbfbfbfbfbfbfbfbfbwbwbwwwwwwwwww
    #    fbfbfbfbfbfbfbfbfbfbfbfbwwwwwwwwwwww
    # Returns the order index of each node on its own stage
    def manual_order(
        self, allow_bubble_before_first_b=False, prioritize_b=False, no_bubble_greedy=True
    ):
        order = [0] * self.nnodes
        f = [0] * self.nstages
        b = [0] * self.nstages
        w = [0] * self.nstages
        o = [0] * self.nstages
        m = [0] * self.nstages
        e = [0] * self.nstages
        t = [0] * self.nnodes
        max_mem = self.config.max_mem or [
            self.get_mem(self.get_id(0, stage, 0)) * self.nmb * 3 for stage in range(self.nstages)]
        comm = self.config.cost_comm
        order_str = [""] * self.nstages
        stage_bubble = [0] * self.nstages

        def get_max_bubble():
            max_bubble = 0
            for bb in stage_bubble:
                max_bubble = max(max_bubble, bb)
            return max_bubble

        def put(stage_j, type_k):
            if type_k == 0:
                _i = f[stage_j]
            elif type_k == 1:
                _i = b[stage_j]
            else:
                _i = w[stage_j]
            _j = stage_j
            _id = self.get_id(type_k, _j, _i)
            _mem = self.get_mem(_id)
            _cost = self.get_cost(_id)
            # TODO
            # assert m[_j] + _mem <= max_mem[stage_j]

            tmp = e[_j] + _cost
            no_bubble = tmp
            if _j > 0 and type_k == 0:
                tmp = max(tmp, t[self.get_id(0, _j - 1, _i)] + comm + _cost)
            if _j < self.nstages - 1 and type_k == 1:
                tmp = max(tmp, t[self.get_id(1, _j + 1, _i)] + comm + _cost)
            if f[_j] > 0:
                stage_bubble[_j] += tmp - no_bubble
            e[_j] = tmp
            t[_id] = tmp
            m[_j] += _mem
            order[_id] = o[_j]
            if type_k == 0:
                f[_j] += 1
            elif type_k == 1:
                b[_j] += 1
            else:
                w[_j] += 1
            o[_j] += 1
            fbw = "fbw"
            order_str[stage_j] += fbw[type_k]

        for i in range(self.nmb):
            if i == 0:
                for j in range(self.nstages):
                    put(j, 0)
                f_required = [0] * self.nstages
                last_t = 0
                for j in range(self.nstages - 1, -1, -1):
                    if j == self.nstages - 1:
                        last_t = t[self.get_id(0, j, i)] + self.get_cost(self.get_id(1, j, i))
                        continue
                    mem = m[j]
                    cost = e[j]
                    while True:
                        f_id = self.get_id(0, j, f[j] + f_required[j])
                        if f[j] + f_required[j] < self.nmb and mem + self.get_mem(f_id) <= max_mem[j]:
                            if allow_bubble_before_first_b:
                                if cost + self.get_cost(f_id) > last_t + comm:
                                    break
                            else:
                                if cost >= last_t + comm:
                                    break
                            mem += self.get_mem(f_id)
                            cost += self.get_cost(f_id)
                            f_required[j] += 1
                        else:
                            break
                    last_t = max(cost, last_t + comm) + self.get_cost(self.get_id(1, j, i))
                for j in range(self.nstages):
                    while j > 0 and f_required[j] > 0 and f_required[j] >= f_required[j - 1] and f[j] + f_required[
                        j] < self.nmb:
                        f_required[j] -= 1
                for j in range(self.nstages):
                    for _ in range(f_required[j]):
                        put(j, 0)
                for j in range(self.nstages - 1, -1, -1):
                    put(j, 1)
                continue
            f_required = [0] * self.nstages
            for j in range(self.nstages):
                if f[j] >= self.nmb:
                    continue
                if j + 1 < self.nstages and f[j] >= f[j + 1] + 2 and prioritize_b:
                    next_plus_fw = (
                        e[j + 1]
                        + self.get_cost(self.get_id(0, j + 1, f[j + 1]))
                        + self.get_cost(self.get_id(1, j + 1, b[j + 1]))
                        + comm
                    )
                    if e[j] >= next_plus_fw:
                        continue
                    f_id = self.get_id(0, j, f[j])
                    f_mem = self.get_mem(f_id)
                    w_cost, w_cnt = 0, 0
                    mem_with_w = m[j] + f_mem
                    while mem_with_w > max_mem[j] and w[j] + w_cnt < b[j]:
                        w_id = self.get_id(2, j, w[j] + w_cnt)
                        w_cost += self.get_cost(w_id)
                        mem_with_w += self.get_mem(w_id)
                        w_cnt += 1
                    if e[j] + self.get_cost(f_id) + w_cost <= next_plus_fw:
                        f_required[j] = 1
                        continue

                    w_cost, w_cnt = 0, 0
                    # mem_with_w = m[j]
                    # while w[j] + w_cnt < b[j]:
                    #     w_id = self.get_id(2, j, w[j] + w_cnt)
                    #     w_cost += self.get_cost(w_id)
                    #     mem_with_w += self.get_mem(w_id)
                    #     w_cnt += 1
                    # if e[j] + w_cost >= next_plus_fw:
                    #     continue
                    if next_plus_fw - (e[j] + w_cost) <= get_max_bubble() - stage_bubble[j]:
                        # TODO: can sample here
                        continue
                f_required[j] = 1
            for j in range(self.nstages - 2, -1, -1):
                f_required[j] = min(f_required[j], f_required[j + 1])
            for j in range(self.nstages):
                if f_required[j] == 0:
                    continue
                f_id = self.get_id(0, j, f[j])
                mem = self.get_mem(f_id)
                while m[j] + mem > max_mem[j]:
                    if w[j] >= b[j]:
                        raise ValueError("Cannot fit memory")
                    put(j, 2)
                if j > 0:
                    while (
                        w[j] < b[j]
                        and e[j] + self.get_cost(self.get_id(2, j, w[j]))
                        <= t[self.get_id(0, j - 1, f[j])] + comm
                    ):
                        put(j, 2)
                    if w[j] < b[j] and e[j] < t[self.get_id(0, j - 1, f[j])] + comm:
                        # TODO: e[j] + self.get_cost(self.get_id(2, j, w[j])) > t[self.get_id(0, j - 1, f[j])] + comm
                        if (
                            t[self.get_id(0, j - 1, f[j])] + comm - e[j]
                            <= get_max_bubble() - stage_bubble[j]
                        ):
                            # TODO: can sample here
                            if no_bubble_greedy:
                                put(j, 2)
                        else:
                            put(j, 2)
                put(j, 0)
            for j in range(self.nstages - 1, -1, -1):
                assert b[j] == i
                b_id = self.get_id(1, j, b[j])
                mem = self.get_mem(b_id)
                while m[j] + mem > max_mem[j]:
                    if w[j] >= b[j]:
                        raise ValueError("Cannot fit memory")
                    put(j, 2)
                if j + 1 < self.nstages:
                    while (
                        w[j] < b[j]
                        and e[j] + self.get_cost(self.get_id(2, j, w[j]))
                        <= t[self.get_id(1, j + 1, i)] + comm
                    ):
                        put(j, 2)
                    if w[j] < b[j] and e[j] < t[self.get_id(1, j + 1, i)] + comm:
                        # TODO: e[j] + self.get_cost(self.get_id(2, j, w[j])) > t[self.get_id(1, j + 1, i)] + comm
                        if (
                            t[self.get_id(1, j + 1, i)] + comm - e[j]
                            <= get_max_bubble() - stage_bubble[j]
                        ):
                            # TODO: can sample here
                            if no_bubble_greedy:
                                put(j, 2)
                        else:
                            put(j, 2)
                if j == 0 and f[j] == self.nmb:
                    while w[j] < b[j]:
                        put(j, 2)
                put(j, 1)

        for i in range(self.nstages):
            while w[i] < self.nmb:
                put(i, 2)
            # print(f"{' ' * i}{order_str[i]}  -> {e[i]}")

        for i in range(self.nstages):
            for j in range(self.nmb):
                f_id = self.get_id(0, i, j)
                b_id = self.get_id(1, i, j)
                w_id = self.get_id(2, i, j)
                f_cost = self.get_cost(f_id)
                b_cost = self.get_cost(b_id)
                w_cost = self.get_cost(w_id)
                assert t[b_id] >= t[f_id] + b_cost
                assert t[w_id] >= t[b_id] + w_cost, f"{i}-{j}, {t[w_id]} >= {t[b_id]} + {w_cost}"
                if i > 0:
                    assert t[f_id] >= t[self.get_id(0, i - 1, j)] + comm + f_cost, f"{i}-{j}"
                if i < self.nstages - 1:
                    assert t[b_id] >= t[self.get_id(1, i + 1, j)] + comm + b_cost

        # print(order)
        best_time = 0
        for i in range(self.nstages):
            time_i = (
                t[self.get_id(2, i, self.nmb - 1)]
                - t[self.get_id(0, i, 0)]
                + self.get_cost(self.get_id(0, i, 0))
            )
            best_time = max(best_time, time_i)

        return order, t, best_time


def initial_solution(graph, print_result=True):
    best_time, order, complete_time = None, None, None
    for allow_bubble_before_first_b in [True, False]:
        for prioritize_b in [True, False]:
            for no_bubble_greedy in [True, False]:
                order_t, complete_time_t, best_time_t = graph.manual_order(
                    allow_bubble_before_first_b=allow_bubble_before_first_b,
                    prioritize_b=prioritize_b,
                    no_bubble_greedy=no_bubble_greedy,
                )
                if best_time is None or best_time_t < best_time:
                    best_time = best_time_t
                    order = order_t
                    complete_time = complete_time_t

    if print_result:
        print_detail(graph, complete_time)
        print("-" * 20, best_time, "-" * 20)
    return best_time, order, complete_time


def print_detail(graph, F):
    typenames = ['F', 'B', 'W']
    times = []
    for stage in range(graph.nstages):
        stage_str = ['.'] * int(F[graph.get_id(2, stage, graph.nmb - 1)] / graph.config.print_scaling)
        for _type in range(3):
            for _mb in range(graph.nmb):
                _id = graph.get_id(_type, stage, _mb)
                end = int(F[_id] / graph.config.print_scaling)
                start = int((F[_id] - graph.get_cost(_id)) / graph.config.print_scaling)
                for j in range(start, end):
                    if j == start or j == end - 1:
                        stage_str[j] = typenames[_type]
                    elif j == start + 1:
                        if _mb >= 10:
                            stage_str[j] = str(_mb // 10)
                        else:
                            stage_str[j] = str(_mb)
                    elif j == start + 2 and _mb >= 10:
                        stage_str[j] = str(_mb % 10)
                    else:
                        stage_str[j] = "-"
        _str = ""
        for _c in stage_str:
            _str += _c
        times.append(
            F[graph.get_id(2, stage, graph.nmb - 1)]
            - F[graph.get_id(0, stage, 0)]
            + graph.get_cost(graph.get_id(0, stage, 0))
        )
        print(_str)
    print('Longest stage time: ', max(times))


def create_schedule(config: GraphConfig, print_result=False):
    graph = Graph.build_graph(config.n_stages, config.n_micro, config)
    best_time, order, complete_time = initial_solution(graph, print_result)
    return create_scheduled_nodes(graph, complete_time)


def create_scheduled_nodes(graph, completion_time):
    typenames = [FuncType.F, FuncType.B, FuncType.W]
    cats = {
        FuncType.F: 0,
        FuncType.B: 1,
        FuncType.W: 2,
    }
    local_order = []
    end_time = []
    for t in completion_time:
        end_time.append(pulp.value(t))
    for stage in range(graph.nstages):
        order = []
        for cat in range(3):
            for mb in range(graph.nmb):
                order.append(
                    ScheduledNode(
                        type=typenames[cat],
                        stage=stage,
                        microbatch=mb,
                        layer_group_idx=stage,
                    )
                )
        order = sorted(order, key=lambda n: completion_time[graph.get_id(cats[n.type], n.stage, n.microbatch)])
        local_order.append(order)
    return local_order
