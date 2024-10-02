# Implementation of vhalf and vmin schedules of Pipeline Parallelism
# with Controllable Memory (https://arxiv.org/abs/2405.15362)
# The reordering is based on a greedy algorithm.
from megatron.core.pipeline_parallel.zerobubble.scheduler.graph import ScheduledNode, B, W, F, FuncType

names = 'FfBbWw'


class PipelineGraph(object):

    def __init__(self, n_stage, n_micro, mem_config,
                 f_cost, b_cost, w_cost, c_cost):
        self.n_stage = n_stage
        self.n_micro = n_micro
        self.mem_config = mem_config
        self.f_cost = f_cost
        self.b_cost = b_cost
        self.w_cost = w_cost
        self.c_cost = c_cost
        self.fbw_cost = [f_cost, b_cost, w_cost]

    def stable_pattern_v_min(self, num_stages):
        interval = 2 if num_stages % 3 == 0 else 0
        schedule = []
        for i in range(num_stages):
            schedule.append([i,
                             num_stages * 2 - i - 1,
                             num_stages * 2 + interval + i,
                             num_stages * 4 + interval - i - 1
                             ])
        return schedule

    def stable_pattern_v_half(self, num_stages):
        interval = 3 if num_stages % 2 == 0 else 0
        schedule = []
        for i in range(num_stages):
            schedule.append([i * 2,
                             num_stages * 3 - i - 2,
                             num_stages * 3 + interval + i * 2 - 1,
                             num_stages * 6 + interval - i - 2
                             ])
        return schedule

    def put_w(self, schedule, split_w=False):
        assert (len(schedule) == 4)
        bound = max([x[-1] for x in schedule])
        inf = bound + 1
        schedule.append([])
        if split_w:
            schedule.append([])
        w_cnt = [0, 0]
        p = [0 for _ in schedule]

        for i in range(bound + 1):
            next_time, next_type = min([(schedule[t][p[t]] if p[t] < len(schedule[t]) else inf, t) for t in range(4)])
            assert next_time != inf
            if next_time == i:
                p[next_type] += 1
                if 2 <= next_type < 4:
                    w_cnt[next_type - 2 if split_w else 0] += 1
            else:
                for w in range(2):
                    if w_cnt[w]:
                        schedule[w + 4].append(i)
                        w_cnt[w] -= 1
                        break
        for w in range(2):
            while w_cnt[w] > 0:
                i += 1
                schedule[w + 4].append(i)
                w_cnt[w] -= 1
        return schedule

    def mem_of_stage(self, schedule):
        memory = [1, 1, 0, 0, -1]
        has_w = len(schedule) > 4
        bound = max([x[-1] for x in schedule])
        inf = bound + 1
        w_cnt = 0
        cur = 0
        res = 0
        p = [0 for _ in schedule]
        for i in range(bound + 1):
            next_time, next_type = min(
                [(schedule[t][p[t]] if p[t] < len(schedule[t]) else inf, t) for t in range(len(schedule))])
            assert next_time != inf
            if next_time == i:
                p[next_type] += 1
                cur += memory[next_type]
                if 2 <= next_type < 4:
                    w_cnt += 1
            elif not has_w:
                if w_cnt:
                    cur += memory[4]
                    w_cnt -= 1
            res = max(res, cur)
        return res

    # Also known as 'squeeze' in the paper.
    def eager_execution_time(self, stage_result, cost, c_cost):
        num_types = len(cost)
        assert num_types == len(stage_result[0])
        num_stages = len(stage_result)
        p = [[0 for _ in range(num_types)] for __ in range(num_stages)]
        total_elements = num_stages * sum([len(t) for t in stage_result[0]])
        end_time = {}
        stage_order = [[] for _ in range(num_stages)]
        inf = max([max([max(type) for type in stage]) for stage in stage_result]) + 1
        for i in range(total_elements):
            next_time, next_stage, next_type, next_mb = min(
                [((stage_result[t // num_types][t % num_types][p[t // num_types][t % num_types]] if p[t // num_types][
                                                                                                        t % num_types] < len(
                    stage_result[t // num_types][t % num_types]) else inf), t // num_types, t % num_types,
                  p[t // num_types][t % num_types]) for t in range(num_types * num_stages)])
            assert next_time != inf
            p[next_stage][next_type] += 1

            deps = 0
            if next_type in {0, 2}:
                if next_stage > 0:
                    deps = max(deps, end_time[(next_stage - 1, next_type, next_mb)] + c_cost)
            if next_type in {1, 3}:
                if next_stage + 1 < num_stages:
                    deps = max(deps, end_time[(next_stage + 1, next_type, next_mb)] + c_cost)
            if 4 > next_type > 0:
                deps = max(deps, end_time[(next_stage, next_type - 1, next_mb)])
            if next_mb > 0:
                deps = max(deps, end_time[(next_stage, next_type, next_mb - 1)])
            if stage_order[next_stage]:
                deps = max(deps, end_time[(next_stage, *stage_order[next_stage][-1])])
            deps += cost[next_type]
            end_time[(next_stage, next_type, next_mb)] = deps
            stage_order[next_stage].append((next_type, next_mb))

        node_time = []
        for stage, stage_content in enumerate(stage_result):
            node_time.append([])
            for type, type_content in enumerate(stage_content):
                node_time[-1].append([])
                for mb, _ in enumerate(type_content):
                    node_time[-1][-1].append(end_time[(stage, type, mb)] - cost[type])

        return max(
            [end_time[(s, num_types - 1, len(stage_result[s][num_types - 1]) - 1)] - end_time[(s, 0, 0)] + cost[0] for s
             in range(num_stages)]), node_time, stage_order

    def reorder(self, stage_result, consider_w):
        num_types = 4
        if consider_w:
            stage_result = [self.put_w(schedule) for schedule in stage_result]
            num_types = 5
        assert num_types == len(stage_result[0])

        current_mem = max([self.mem_of_stage(s) for s in stage_result])
        num_stages = len(stage_result)

        p = [[0 for _ in range(num_types)] for __ in range(num_stages)]
        occupied = [['' for x in range(stage[-1][-1] + 1)] for stage in stage_result]
        total_elements = num_stages * sum([len(t) for t in stage_result[0]])

        phase = 0
        inf = max([max([max(type) for type in stage]) for stage in stage_result]) + 1
        for i in range(total_elements):
            next_time, next_stage, next_type, next_mb = min(
                [((stage_result[t // num_types][t % num_types][p[t // num_types][t % num_types]] if p[t // num_types][
                                                                                                        t % num_types] < len(
                    stage_result[t // num_types][t % num_types]) else inf), t // num_types, t % num_types,
                  p[t // num_types][t % num_types]) for t in range(num_types * num_stages)])
            assert next_time != inf

            p[next_stage][next_type] += 1

            if next_type == 3 and next_stage == 0 and phase == 0:
                phase += 1
            if phase == 1:
                if next_type == 0 and next_mb == len(stage_result[0][0]) - 1:
                    phase += 1
                occupied[next_stage][stage_result[next_stage][next_type][next_mb]] = f'{names[next_type]}{next_mb}'
                continue
            if phase == 0 and not consider_w:
                occupied[next_stage][stage_result[next_stage][next_type][next_mb]] = f'{names[next_type]}{next_mb}'
                continue

            deps = -1
            if next_type in {0, 2}:
                if next_stage > 0:
                    deps = max(deps, stage_result[next_stage - 1][next_type][next_mb])
            if next_type in {1, 3}:
                if next_stage + 1 < num_stages:
                    deps = max(deps, stage_result[next_stage + 1][next_type][next_mb])
            if 4 > next_type > 0:
                deps = max(deps, stage_result[next_stage][next_type - 1][next_mb])
            if 4 == next_type:
                b_time = sorted(stage_result[next_stage][2] + stage_result[next_stage][3])[next_mb]
                deps = max(deps, b_time)
            if next_mb > 0:
                deps = max(deps, stage_result[next_stage][next_type][next_mb - 1])

            deps += 1
            assert deps <= next_time
            ok = False
            for possible_time in range(deps, next_time):
                if not occupied[next_stage][possible_time]:
                    stage_result[next_stage][next_type][next_mb] = possible_time
                    if self.mem_of_stage(stage_result[next_stage]) <= current_mem:
                        ok = True
                        break
            if not ok:
                stage_result[next_stage][next_type][next_mb] = next_time
            occupied[next_stage][stage_result[next_stage][next_type][next_mb]] = f'{names[next_type]}{next_mb}'

        if not consider_w:
            stage_result = [self.put_w(schedule) for schedule in stage_result]

        return self.eager_execution_time(stage_result, [1, 1, 1, 1, 1], 0)[1]

    def schedule_from_pattern(self, schedule, nmb, cost, do_reorder=True):
        stage_result = [[list(range(x, x + 6 * nmb, 6)) for x in stage] for stage in schedule]
        if not do_reorder:
            stage_result = [self.put_w(schedule, split_w=True) for schedule in stage_result]
            return self.eager_execution_time(stage_result, cost, self.c_cost)
        r1 = self.reorder(stage_result, consider_w=True)

        # Remove w and do squeezing again
        r1 = [schedule[:4] for schedule in r1]
        r2 = self.reorder(r1, consider_w=False)

        # Remove w decisions and split w into two chunks. Redo execution time eval.
        r2 = [schedule[:4] for schedule in r2]
        r2 = [self.put_w(schedule, split_w=True) for schedule in r2]
        # print(r2)
        return self.eager_execution_time(
            r2, cost, c_cost=self.c_cost)

    def to_csv(self, stage_result):
        for schedule in stage_result:
            r = [''] * (schedule[-1][-1] + 1)
            for type, order in enumerate(schedule):
                for mb, o in enumerate(order):
                    assert not r[o], f'{r[o]}, {names[type]}{mb}'
                    r[o] = f'{names[type]}{mb}'
            print(','.join(r))

    def create_schedule(self, config):
        schedulefunc = {
            'min': self.stable_pattern_v_min,
            'half': self.stable_pattern_v_half,
        }
        max_time, start_time, stage_order = self.schedule_from_pattern(
            schedulefunc[self.mem_config](self.n_stage), self.n_micro,
            [self.fbw_cost[x // 2] for x in range(len(self.fbw_cost) * 2)],
            do_reorder=True)

        expected_time = sum(self.fbw_cost) * self.n_micro * 2
        # # self.print_details(end_time, print_scaling=1)
        bubble_rate = (max_time - expected_time) / max_time
        print("%2d %3d, [%5d %5d %5d %5d], %s -> %6.4f" % \
              (self.n_stage, self.n_micro, *self.fbw_cost, self.c_cost, self.mem_config, bubble_rate))

        local_order = [[] for _ in range(self.n_stage)]
        for stage in range(self.n_stage):
            for (type, _micro_) in stage_order[stage]:
                _cat_ = type // 2
                _chunk_ = type % 2

                chunk_index = _chunk_
                chunk = _chunk_ if _cat_ == 0 \
                    else config.max_chunks - 1 - _chunk_
                if _cat_ in (0, 1):
                    assert config.max_chunks == 2
                else:
                    assert _cat_ == 2
                layer_group_idx = config.n_stages * chunk
                if chunk % 2 == 0:
                    layer_group_idx += stage
                else:
                    layer_group_idx += config.n_stages - 1 - stage
                local_order[stage].append(ScheduledNode(
                    type=[F, B, W][_cat_],
                    chunk=chunk,
                    stage=stage,
                    microbatch=_micro_,
                    layer_group_idx=layer_group_idx,
                ))
        return local_order
