from collections import Counter

from megatron.core.pipeline_parallel.zerobubble.scheduler.graph import GraphConfig, ScheduledNode, F, BW

pattern_size = 6


def transform_schedule(schedule, f, b, w, c):
    result = []

    stage_order = []
    local_prev = {}
    stages = len(schedule)

    for sid, stage in enumerate(schedule):
        counter = Counter()
        order = []
        for p in stage:
            if not p.strip():
                continue
            mb = counter.get(p, 0)
            if order:
                local_prev[(sid, p, mb)] = order[-1]
            order.append((p, mb))
            counter.update(p)
        stage_order.append(order)
    nmb = max(counter.values())
    time_map = {}
    cost = {
        'F': f,
        'B': b + w,
        'f': f,
        'b': b + w,
    }

    def get_time(stage, type, mb):
        if (stage, type, mb) in time_map:
            return time_map.get((stage, type, mb))
        time = 0
        if (stage, type, mb) in local_prev:
            time = get_time(stage, *local_prev[(stage, type, mb)])
        if type in "FB" and stage > 0:
            time = max(time, get_time(stage - 1, type, mb) + c)
        if type in "fb" and stage + 1 < len(schedule):
            time = max(time, get_time(stage + 1, type, mb) + c)
        time_map[(stage, type, mb)] = time + cost[type]
        return time_map[(stage, type, mb)]

    r = 0
    for sid, stage in enumerate(schedule):
        r = max(get_time(sid, 'b', nmb - 1) - get_time(sid, 'F', 0) + f, r)

    for sid, stage in enumerate(stage_order):
        result_stage = []
        for p, mb in stage:
            chunk = 1 if p in "fBW" else 0
            assert p.upper() != 'W'
            func_type = F if p.upper() == 'F' else BW
            layer_group_idx = stages * chunk
            if chunk % 2 == 0:
                layer_group_idx += sid
            else:
                layer_group_idx += stages - 1 - sid
            result_stage.append(ScheduledNode(
                type=func_type,
                chunk=chunk,
                stage=sid,
                microbatch=mb,
                layer_group_idx=layer_group_idx,
                # get_time(sid, p, mb) - cost[p],
                # get_time(sid, p, mb)
            ))
        result.append(result_stage)
    return result


def get_pattern_str(pos):
    pattern = [" "] * pattern_size
    notations = "FfBbWw"
    for i, v in enumerate(pos):
        if v < 0:
            continue
        pattern[v] = notations[i]
    _str = ""
    for v in pattern:
        _str += v
    return _str


def init_repeated_schedule(p, m, patterns):
    repeated = []
    _len = 4 * p + m + 1
    for i in range(p):
        str_i = get_pattern_str(patterns[i]) * _len
        repeated_i = []
        for v in str_i:
            repeated_i.append(v)
        repeated.append(repeated_i)
    return repeated


def clear_invalid(repeated, stage, pos, offset=-1):
    while 0 <= pos < len(repeated[stage]):
        repeated[stage][pos] = ' '
        pos += offset * pattern_size
    return repeated


def clear_invalid_index(repeated, m):
    p = len(repeated)
    index = pattern_size
    for identifier in "FfBb":
        if identifier in "FB":
            _iter = range(p)
        else:
            _iter = range(p - 1, -1, -1)
        for i in _iter:
            for j in range(pattern_size):
                if repeated[i][index] == identifier:
                    clear_invalid(repeated, i, index - pattern_size, offset=-1)
                    clear_invalid(repeated, i, index + pattern_size * m, offset=1)
                    index += 1
                    if identifier in "Bb":
                        w_identifier = {'B': 'W', 'b': 'w'}[identifier]
                        for k in range(pattern_size):
                            if repeated[i][index + k] == w_identifier:
                                clear_invalid(repeated, i, index + k - pattern_size, offset=-1)
                                clear_invalid(repeated, i, index + k + pattern_size * m, offset=1)
                                break
                    break
                index += 1
    return repeated


def process_warmup_without_increasing_peak_mem(schedules, m):
    peak_mem = 0
    mem = [[0 for _ in range(len(schedules[0]))] for _ in range(len(schedules))]
    loc = [[{key: -1 for key in ('F', 'f', 'B', 'b', 'W', 'w')} for _ in range(m + 2)] for _ in range(len(schedules))]
    cntr = [{key: 0 for key in ('F', 'f', 'B', 'b', 'W', 'w')} for _ in range(len(schedules))]
    for sid in range(len(schedules)):
        cur = 0
        for i in range(len(schedules[sid])):
            if schedules[sid][i] in "Ff":
                cur += 1
            if schedules[sid][i] in "Ww":
                cur -= 1
            mem[sid][i] = cur
            peak_mem = max(peak_mem, cur)

    for i in range(len(schedules[0])):
        for sid in range(len(schedules)):
            if schedules[sid][i] == ' ':
                continue
            cntr[sid][schedules[sid][i]] += 1
            cnt = cntr[sid][schedules[sid][i]]
            pos = -1
            if cnt > 1:
                pos = loc[sid][cnt - 1][schedules[sid][i]]
            if schedules[sid][i] == 'W':
                pos = max(pos, loc[sid][cnt]['B'])
            if schedules[sid][i] == 'w':
                pos = max(pos, loc[sid][cnt]['b'])
            if schedules[sid][i] == 'F' and sid > 0:
                pos = max(pos, loc[sid - 1][cnt]['F'])
            if schedules[sid][i] == 'f':
                if sid != len(schedules) - 1:
                    pos = max(pos, loc[sid + 1][cnt]['f'])
                else:
                    pos = max(pos, loc[sid][cnt]['F'])
            if schedules[sid][i] == 'B':
                if sid != 0:
                    # Because B and W are always combined
                    pos = max(pos, loc[sid - 1][cnt]['W'])
                else:
                    pos = max(pos, loc[sid][cnt]['f'])
            if schedules[sid][i] == 'b':
                if sid != len(schedules) - 1:
                    # Because B and W are always combined
                    pos = max(pos, loc[sid + 1][cnt]['w'])
                else:
                    pos = max(pos, loc[sid][cnt]['W'])
            pos += 1
            while schedules[sid][pos] != ' ' and pos < i:
                pos += 1
            if schedules[sid][i] in "Bb":
                while pos < i and (schedules[sid][pos] != ' ' or schedules[sid][pos + 1] != ' '):
                    pos += 1
            if pos == i:
                loc[sid][cnt][schedules[sid][i]] = i
                continue
            if schedules[sid][i] in "BbWw":
                schedules[sid][pos] = schedules[sid][i]
                schedules[sid][i] = ' '
                if schedules[sid][pos] in "Ww":
                    for j in range(pos, i):
                        mem[sid][j] -= 1
                loc[sid][cnt][schedules[sid][pos]] = pos
                continue

            # If F or f:

            place = i
            while place > pos and mem[sid][place - 1] < peak_mem:
                place -= 1
            while place < i and schedules[sid][place] != ' ':
                place += 1
            if place == i:
                loc[sid][cnt][schedules[sid][i]] = i
                continue
            pos = place
            schedules[sid][pos] = schedules[sid][i]
            schedules[sid][i] = ' '
            for j in range(pos, i):
                mem[sid][j] += 1
            loc[sid][cnt][schedules[sid][pos]] = pos
    return schedules


def schedule_by_pattern(p, m, patterns):
    schedules = init_repeated_schedule(p, m, patterns)
    schedules = clear_invalid_index(schedules, m)

    schedules = process_warmup_without_increasing_peak_mem(schedules, m)
    for sid in range(len(schedules)):
        cnt = {_id: 0 for _id in "FfBbWw"}
        for i in range(len(schedules[sid])):
            if (schedules[sid][i] == ' '):
                continue
            if cnt[schedules[sid][i]] >= m:
                schedules[sid][i] = ' '
            else:
                cnt[schedules[sid][i]] += 1

    return schedules


def create_whole_pattern(p):
    whole_pattern = [[0 for _ in range(6)] for _ in range(p)]
    now = 0
    for i in range(p):
        now += 1
        whole_pattern[i][0] = now
    for i in range(p):
        now += 1
        whole_pattern[p - 1 - i][1] = now
    now += 1
    if p % 3 == 0:
        now += 3
    cyc = (3 - (p + 2) % 3) % 3
    for i in range(p):
        whole_pattern[i][2], whole_pattern[i][4] = now, now + 1
        cyc += 1
        now += 2
        if (cyc == 3):
            cyc = 0
            now += 3
    for i in range(p):
        whole_pattern[p - 1 - i][3], whole_pattern[p - 1 - i][5] = now, now + 1
        cyc += 1
        now += 2
        if (cyc == 3):
            cyc = 0
            now += 3
    for sid in range(p):
        for i in range(6):
            whole_pattern[sid][i] %= 6
    return whole_pattern


def schedule(p, m, cost):
    whole_pattern = create_whole_pattern(p)
    s = schedule_by_pattern(p, m, whole_pattern)
    for sid in range(len(s)):
        for i in range(len(s[sid])):
            if s[sid][i] in "Ww":
                s[sid][i] = ' '
    res = transform_schedule(s, *cost)
    return res


def create_schedule(config: GraphConfig):
    cost = [config.cost_f[0], config.cost_b[0], config.cost_w[0], config.cost_comm]
    res = schedule(config.n_stages, config.n_micro, cost)
    return res
