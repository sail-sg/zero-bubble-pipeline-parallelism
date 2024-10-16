pattern_size = 6
from collections import Counter, deque
from dataclasses import dataclass


@dataclass(eq=True, frozen=True)
class ScheduledNode:
    type: str
    chunk: int
    stage: int
    minibatch: int
    start_time: int
    completion_time: int


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
        'B': b,
        'W': w,
        'f': f,
        'b': b,
        'w': w,
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
        # print(f'{stage} {type}:{mb}', time + cost[type])
        time_map[(stage, type, mb)] = time + cost[type]
        return time_map[(stage, type, mb)]

    r = 0
    for sid, stage in enumerate(schedule):
        r = max(get_time(sid, 'W', nmb - 1) - get_time(sid, 'F', 0) + f, r)
        r = max(get_time(sid, 'w', nmb - 1) - get_time(sid, 'F', 0) + f, r)

    for sid, stage in enumerate(stage_order):
        result_stage = []
        for p, mb in stage:
            result_stage.append(ScheduledNode(
                p.upper(),
                p in "fBW",
                sid,
                mb,
                get_time(sid, p, mb) - cost[p],
                get_time(sid, p, mb)
            )
            )
        result.append(result_stage)
    return result


def evaluate_schedule(schedule, f, b, w, c):
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
        'B': b,
        'W': w,
        'f': f,
        'b': b,
        'w': w,
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
        # print(f'{stage} {type}:{mb}', time + cost[type])
        time_map[(stage, type, mb)] = time + cost[type]
        return time_map[(stage, type, mb)]

    r = 0
    for sid, stage in enumerate(schedule):
        r = max(get_time(sid, 'W', nmb - 1) - get_time(sid, 'F', 0) + f, r)
        r = max(get_time(sid, 'w', nmb - 1) - get_time(sid, 'F', 0) + f, r)
    return r


debug = False


def print_schedules(schedules, msg=None, force=False):
    if not debug and not force:
        return
    if msg is not None:
        print(msg)
    for seq in schedules:
        _str = ""
        for v in seq:
            _str += v
        print(_str)


def get_building_block_str(pos):
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


def get_peak_mem(schedules, return_all=False):
    max_peak = 0
    all_peak = []
    for schedule_ in schedules:
        peak, mem = 0, 0
        for v in schedule_:
            if v in "Ff":
                mem += 1
            elif v in "Ww":
                mem -= 1
            peak = max(peak, mem)
        all_peak.append(peak)
        max_peak = max(max_peak, peak)
    if return_all:
        return all_peak
    return max_peak


def calc_bubble(schedules):
    stage_bubbles = []
    for i in range(len(schedules)):
        max_len = 0
        count = 0
        for j in range(len(schedules[i])):
            if schedules[i][j] != ' ':
                max_len = j + 1
                count += 1
        stage_bubbles.append(max_len - count - i)
    return stage_bubbles


def init_repeated_schedule(p, m, building_block):
    repeated = []
    _len = 4 * p + m + 1
    for i in range(p):
        str_i = get_building_block_str(building_block[i]) * _len
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
    """
    FFFFFFFFFF     fBWfBWfBWfBWfBW  b
     FFFFFFFFF    f fBWfBWfBWfBWFBWb
      FFFFFFFF   f f fBWfBWfBWFBW b
       FFFFFFF  f f f fBWfBWFBW Bb
        FFFFFF f f f f fBWFBWFBWb
         FFFFFfFf f f f  BWFBW b
          FFFfFfFfFf f    BW Bb
           FfFfFfFfFfF     BWb
    We reorganize the warmup phase in the following way (i -> pipeline stage from 0):
        1. Before the first B, we set #f = min(i+1, peak_mem//2), #F = peak_mem - #f
        2. Before the first b, #f = peak_mem//2
        3. The offset between the first B is 1
        4. Before the first b, we use the pattern of (BWf)*j + (BWF)*k,
           where j = max(0, peak_mem//2 - (i+1)), k = max(0, #W - j - 1)
    """
    # process warmup phase (before the first b)
    p = len(schedules)
    peak_mem = get_peak_mem(schedules)
    peak_mem = min(peak_mem, 2 * p)
    cnt_f, cnt_ff = [], []
    for i in range(p):
        cc_ff = min(i + 1, peak_mem // 2)
        cc_ff = min(cc_ff, m)
        cc_f = min(peak_mem - cc_ff, m)
        cnt_f.append(cc_f)
        cnt_ff.append(cc_ff)
    distance_b2bb = 0
    for j in range(len(schedules[p - 1])):
        if schedules[p - 1][j] == 'B':
            for k in range(j, len(schedules[p - 1])):
                if schedules[p - 1][k] == 'b':
                    distance_b2bb = k - j
                    break
            break
    for i in range(p):
        c_f, c_ff, c_b, c_w = 0, 0, 0, 0
        for j in range(len(schedules[i])):
            char = schedules[i][j]
            if char == 'F':
                c_f += 1
            elif char == 'f':
                c_ff += 1
            elif char == 'B':
                c_b += 1
            elif char == 'W':
                c_w += 1
            elif char == 'b':
                break
                # This logic can be removed because it is too complicated and should not impact the optimal solution
                bj = j
                while j < len(schedules[i]):
                    char = schedules[i][j]
                    if char == 'f' and c_ff < cnt_ff[p - 1]:
                        schedules[i][j] = ' '
                        c_ff += 1
                    if char == 'B' and c_b < c_ff:
                        if c_b < (2 * (p - i) + distance_b2bb) // 3 or c_b < cnt_ff[p - 1] - cnt_ff[i]:
                            # there is empty space, or the number of B is not enough to cover extra f
                            schedules[i][j] = ' '
                            c_b += 1
                    if char == 'W' and c_w < c_b:
                        if c_w < (2 * (p - i) + distance_b2bb - 1) // 3 or c_w < cnt_ff[p - 1] - cnt_ff[i]:
                            # there is empty space, or the number of W is not enough to cover extra f
                            schedules[i][j] = ' '
                            c_w += 1
                    j += 1
                j = bj
                while j < len(schedules[i]):
                    if schedules[i][j] == 'F':
                        if c_f < c_ff or c_f < cnt_f[i] or c_f - cnt_f[i] + c_ff - cnt_ff[i] < c_w - 1:
                            # put enough F, or there are some unused BW
                            schedules[i][j] = ' '
                            c_f += 1
                    j += 1
                break
            else:
                assert char == ' '
            schedules[i][j] = ' '
        # assert c_f >= cnt_f[i] and c_ff >= cnt_ff[i]
        # assert c_w >= cnt_ff[p - 1] - cnt_ff[i] and c_b >= cnt_ff[p - 1] - cnt_ff[i]
        j = i
        u_f, u_ff, u_b, u_w = 0, 0, 0, 0
        for _ in range(2 * (p - 1 - i)):
            if u_f < cnt_f[i] and u_f < c_f:
                schedules[i][j] = 'F'
                u_f += 1
            j += 1
        for _ in range(i + 1):
            if u_f < cnt_f[i] and u_f < c_f:
                schedules[i][j] = 'F'
                u_f += 1
            j += 1
            if u_ff < cnt_ff[i] and u_ff < c_ff:
                schedules[i][j] = 'f'
                u_ff += 1
            j += 1
        while u_f < c_f or u_ff < c_ff or u_b < c_b or u_w < c_w:
            if u_b < c_b:
                schedules[i][j] = 'B'
                u_b += 1
            j += 1
            if u_w < c_w:
                schedules[i][j] = 'W'
                u_w += 1
            j += 1
            if u_ff < c_ff:
                assert u_ff < u_f
                schedules[i][j] = 'f'
                u_ff += 1
            elif u_f < c_f:
                schedules[i][j] = 'F'
                u_f += 1
            j += 1
    return schedules


def squeeze_without_change_order(schedules, m):
    p = len(schedules)
    squeezed = [[' '] * len(schedules[_]) for _ in range(p)]
    max_len = check_and_get_schedule_len(schedules)

    identifier_cnt = [{_id: 0 for _id in "FfBbWw"} for _ in range(p)]
    identifier_index = [{_id: -1 for _id in "FfBbWw"} for _ in range(p * m)]
    stage_index = [0 for _ in range(p)]
    for j in range(max_len):
        for _dir in range(2):
            if _dir == 0:
                _iter = range(p)
            else:
                _iter = range(p - 1, -1, -1)
            for i in _iter:
                identifier = schedules[i][j]
                if identifier == ' ':
                    continue
                if _dir == 0 and identifier in "fbw":
                    continue
                if _dir == 1 and identifier in "FBW":
                    continue
                _cnt = identifier_cnt[i][identifier]
                assert _cnt < m, "{} - {}, {}".format(i, identifier, _cnt)
                if identifier in "Ww" or (i == 0 and identifier in "FB") or (i == p - 1 and identifier in "fb"):
                    if i == 0 and identifier == 'B':
                        assert identifier_index[_cnt * p + i]['f'] >= 0
                    if i == p - 1 and identifier == 'f':
                        assert identifier_index[_cnt * p + i]['F'] >= 0
                    if i == p - 1 and identifier == 'b':
                        assert identifier_index[_cnt * p + i]['B'] >= 0
                    index = stage_index[i]
                elif identifier in "FB":
                    assert identifier_index[_cnt * p + i - 1][identifier] >= 0, "{} {} {}".format(i, identifier, _cnt)
                    index = max(identifier_index[_cnt * p + i - 1][identifier] + 1, stage_index[i])
                elif identifier in "fb":
                    assert identifier_index[_cnt * p + i + 1][identifier] >= 0, "{} {} {}".format(i, identifier, _cnt)
                    index = max(identifier_index[_cnt * p + i + 1][identifier] + 1, stage_index[i])
                else:
                    raise
                squeezed[i][index] = identifier
                identifier_cnt[i][identifier] = _cnt + 1
                identifier_index[_cnt * p + i][identifier] = index
                stage_index[i] = index + 1
    new_len = max(stage_index)
    for i in range(p):
        squeezed[i] = squeezed[i][:new_len]
    return squeezed


def process_cooldown(schedules, m):
    """
           fBW       bwbwbwbw
          fBWBW     bwbwbwbw
         fBWBWBW   bwbwbwbw
        fBWBWBWBW bwbwbwbw
       f  BWBWBWBbWbwbwbww
      f    BWBWBbBbWbWbwwww
     f      BWBbBbBbWbWWwwww
    f        BbBbBbBbWWWWwwww
    We reorganize the cooldown phase in the following way (i -> pipeline stage from 0):
        1. After the last f, we set #b = (peak_mem+1)//2, and #B = min(i+1, peak_mem - #b)
        2. After the last f, we make all the dependencies as tight as possible
    """
    p = len(schedules)

    peak_mem = get_peak_mem(schedules)
    assert peak_mem <= 2 * p, peak_mem
    max_bb = (peak_mem + 1) // 2
    max_bb = min(max_bb, m)
    max_b = min(peak_mem - max_bb, m)

    # 1: reorganize B/b and remove W/w in cooldown phase
    starting_index = -1
    for i in range(p):
        c_b, c_bb, c_w, c_ww = 0, 0, 0, 0
        last_ff_index = -1
        # collect B/b which can be reordered
        for j in range(len(schedules[i]) - 1, -1, -1):
            char = schedules[i][j]
            if char == 'f' and last_ff_index == -1:
                last_ff_index = j
            if char == 'B' and c_b < i + 1 and c_b < max_b:
                schedules[i][j] = ' '
                c_b += 1
            if char == 'b' and c_bb < max_bb:
                schedules[i][j] = ' '
                c_bb += 1
        # clear W in the tail (#W + #w >= peak_mem & #W >= #B & #w >= #b)
        for j in range(len(schedules[i]) - 1, -1, -1):
            char = schedules[i][j]
            if c_w >= c_b and c_ww >= c_bb and c_w + c_ww >= peak_mem:
                break
            if char == 'W':
                schedules[i][j] = ' '
                c_w += 1
            if char == 'w':
                schedules[i][j] = ' '
                c_ww += 1
        if i == 0:
            starting_index = last_ff_index
        # reorganize B/b in the tail
        for k in range(c_bb):
            index = starting_index - i + 2 * p - 2 * k
            assert schedules[i][index] == ' ', "{} {} {}".format(schedules[i][index], k, i)
            schedules[i][index] = 'b'
        for k in range(c_b):
            index = starting_index + 1 + i - 2 * k
            # assert schedules[i][index] == ' ', schedules[i][index]
            schedules[i][index] = 'B'

    # 2: add W back in cooldown phase
    max_len = 0
    for i in range(p):
        c_w, c_ww = 0, 0
        last_w_index = -1
        for j in range(len(schedules[i]) - 1, -1, -1):
            if schedules[i][j] in "Ww":
                last_w_index = j
                break
        for j in range(len(schedules[i])):
            char = schedules[i][j]
            if char == 'B':
                c_w += 1
            elif char == 'b':
                c_ww += 1
            elif char == 'W':
                c_w -= 1
            elif char == 'w':
                c_ww -= 1
            if char == ' ' and j > last_w_index:
                if c_w > 0:
                    schedules[i][j] = 'W'
                    c_w -= 1
                elif c_ww > 0:
                    schedules[i][j] = 'w'
                    c_ww -= 1
        for _ in range(c_w):
            schedules[i].append('W')
        for _ in range(c_ww):
            schedules[i].append('w')
        max_len = max(max_len, len(schedules[i]))
    for i in range(p):
        for _ in range(len(schedules[i]), max_len):
            schedules[i].append(' ')

    schedules = squeeze_without_change_order(schedules, m)
    return schedules


def check_and_get_schedule_len(schedules):
    max_len = 0
    for seq in schedules:
        assert max_len == 0 or max_len == len(seq)
        max_len = max(max_len, len(seq))
    return max_len


def release_w_in_warmup_if_under_memory(schedules, peak_mem=None):
    """
    FF     fBWfBW   bwbw       ->         FF     fBfBWW  bwbw
     FF   f fBW BW bwbw        ->          FF   f fBWBW bwbw
      FF f f  BW BbWbww        ->           FF f f  BWBbWbww
       FfFf    BbWBbwWw        ->            FfFf    BbBbWwWw
    When the number of micro-batches is too small (than mem), the warmup phase is not optimal. We simply remove some
    preceding W to fully utilize the memory to reduce unnecessary bubbles.
    """
    p = len(schedules)
    max_len = check_and_get_schedule_len(schedules)
    all_peak_mem = get_peak_mem(schedules, return_all=True)
    peak_mem = peak_mem or max(all_peak_mem)
    min_peak = min(all_peak_mem)
    for i in range(p):
        cnt = 0
        padding = [" "] * (peak_mem - min_peak)
        for j in range(max_len):
            if all_peak_mem[i] + cnt >= peak_mem:
                break
            if schedules[i][j] in "Ww":
                padding[cnt] = schedules[i][j]
                schedules[i][j] = ' '
                cnt += 1
        schedules[i].extend(padding)
    # max_len += peak_mem - min_peak
    return schedules


def reorder_greedily_without_increasing_peak_mem(schedules, m, starting_index=None, ending_index=None, peak_mem=None):
    """
    We iterate all the cells from left to right. If a vacant cell (which means a bubble) is encountered, we try to
    find a computation pass to fill this bubble. We iterate all the following computation passes in the same device,
    and check whether it is possible to move if we keep all other passes unchanged. If the check succeeds, we move it
    to the vacant cell, and the bubble is filled.
    """
    p = len(schedules)
    if starting_index is not None:
        assert isinstance(starting_index, list) and len(starting_index) == p
    if ending_index is not None:
        assert isinstance(ending_index, list) and len(ending_index) == p

    peak_mem = peak_mem or get_peak_mem(schedules)
    max_len = check_and_get_schedule_len(schedules)
    starting_index = starting_index or [0] * p
    ending_index = ending_index or [max_len] * p
    last_index = [{_id: -1 for _id in "FfBbWw"} for _ in range(p)]
    for i in range(p):
        for j in range(max_len):
            identifier = schedules[i][j]
            if identifier == ' ':
                continue
            last_index[i][identifier] = j

    stage_mem = [0] * p

    def update_mem(stage_i, pass_c):
        if pass_c in "Ff":
            stage_mem[stage_i] += 1
        elif pass_c in "Ww":
            stage_mem[stage_i] -= 1

    identifier_cnt = [{_id: 0 for _id in "FfBbWw"} for _ in range(p)]
    identifier_index = [{_id: -1 for _id in "FfBbWw"} for _ in range(p * m)]
    for j in range(0, max_len):
        for i in range(p):
            identifier = schedules[i][j]
            if identifier in "FfBbWw":
                _cnt = identifier_cnt[i][identifier]
                identifier_cnt[i][identifier] = _cnt + 1
                identifier_index[_cnt * p + i][identifier] = j
                update_mem(i, identifier)
                continue
            assert identifier == ' '
            if j < starting_index[i] or j >= ending_index[i]:
                continue
            available = set()
            for c in "FfBbWw":
                if last_index[i][c] > j:
                    available.add(c)
            mem_delta, peak_delta = 0, 0
            for k in range(j + 1, ending_index[i]):
                if len(available) == 0:
                    break
                identifier = schedules[i][k]
                if identifier in "Ff":
                    mem_delta += 1
                elif identifier in "Ww":
                    mem_delta -= 1
                prev_peak = peak_delta
                peak_delta = max(peak_delta, mem_delta)
                if identifier == ' ' or identifier not in available:
                    continue
                available.remove(identifier)
                if identifier in "Ff" and stage_mem[i] + prev_peak >= peak_mem:
                    # will increase peak memory
                    continue
                can_move = True
                _cnt = identifier_cnt[i][identifier]
                if identifier in "FB":
                    if i > 0:
                        _index = identifier_index[_cnt * p + i - 1][identifier]
                        if _index <= -1 or _index >= j:
                            can_move = False
                    elif identifier == 'B':
                        if identifier_cnt[i]['f'] <= _cnt:
                            can_move = False
                elif identifier in "fb":
                    if i + 1 < p:
                        _index = identifier_index[_cnt * p + i + 1][identifier]
                        if _index <= -1 or _index >= j:
                            can_move = False
                    else:
                        _pi = 'F' if identifier == 'f' else 'B'
                        if identifier_cnt[i][_pi] <= _cnt:
                            can_move = False
                elif identifier in "Ww":
                    _bi = 'B' if identifier == 'W' else 'b'
                    if identifier_cnt[i][_bi] <= _cnt:
                        can_move = False
                else:
                    assert False
                if not can_move:
                    continue
                # if i == 0:
                #     print(peak_mem, stage_mem[i], identifier, mem_delta)
                schedules[i][j] = identifier
                schedules[i][k] = ' '
                identifier_cnt[i][identifier] = _cnt + 1
                identifier_index[_cnt * p + i][identifier] = j
                update_mem(i, identifier)
                break
    return schedules


def check_correctness(schedules, m, raise_exception=False):
    p = len(schedules)
    c_index = [{_id: -1 for _id in "FfBbWw"} for _ in range(p * m)]
    for i in range(p):
        c_cnt = {_id: 0 for _id in "FfBbWw"}
        for j in range(len(schedules[i])):
            c = schedules[i][j]
            if c in "FfBbWw":
                _cnt = c_cnt[c]
                assert _cnt < m
                c_index[_cnt * p + i][c] = j
                c_cnt[c] = _cnt + 1
        for c in "FfBbWw":
            if c_cnt[c] != m:
                assert not raise_exception
                return False
    for i in range(p):
        for j in range(m):
            for c in "FfBbWw":
                if c_index[j * p + i][c] == -1:
                    assert not raise_exception
                    return False
            if c_index[j * p + i]['B'] >= c_index[j * p + i]['W']:
                assert not raise_exception, f"{i} {j} {c}"
                return False
            if c_index[j * p + i]['b'] >= c_index[j * p + i]['w']:
                assert not raise_exception
                return False
            if i == 0:
                if c_index[j * p + i]['f'] >= c_index[j * p + i]['B']:
                    assert not raise_exception
                    return False
            elif i == p - 1:
                if c_index[j * p + i]['F'] >= c_index[j * p + i]['f']:
                    assert not raise_exception
                    return False
                if c_index[j * p + i]['B'] >= c_index[j * p + i]['b']:
                    assert not raise_exception
                    return False
            else:
                if c_index[j * p + i - 1]['F'] >= c_index[j * p + i]['F']:
                    assert not raise_exception
                    return False
                if c_index[j * p + i - 1]['B'] >= c_index[j * p + i]['B']:
                    assert not raise_exception
                    return False
                if c_index[j * p + i + 1]['f'] >= c_index[j * p + i]['f']:
                    assert not raise_exception
                    return False
                if c_index[j * p + i + 1]['b'] >= c_index[j * p + i]['b']:
                    assert not raise_exception
                    return False
    return True


def relabel_w(schedules, m):
    p = len(schedules)
    c_cnt = [{_id: 0 for _id in "FfBbWw"} for _ in range(p)]
    for i in range(p):
        for j in range(len(schedules[i])):
            if schedules[i][j] == ' ':
                continue
            c_cnt[i][schedules[i][j]] += 1
        for c in "FfBbWw":
            assert c_cnt[i][c] == m, f"{i}, {c}, {c_cnt[i][c]}"
    for i in range(p):
        w_queue = deque(maxlen=2 * m)
        for j in range(len(schedules[i])):
            identifier = schedules[i][j]
            if identifier == 'B':
                w_queue.append('W')
            elif identifier == 'b':
                w_queue.append('w')
            elif identifier in "Ww":
                assert len(w_queue) > 0, f"{i} {j}"
                schedules[i][j] = w_queue.popleft()
        assert len(w_queue) == 0
    return schedules


def remove_redundancy(schedules, m):
    for sid in range(len(schedules)):
        cnt = {_id: 0 for _id in "FfBbWw"}
        for i in range(len(schedules[sid])):
            if schedules[sid][i] == ' ':
                continue
            if cnt[schedules[sid][i]] >= m:
                schedules[sid][i] = ' '
            else:
                cnt[schedules[sid][i]] += 1
    return schedules


def schedule_by_building_block(p, m, building_block, max_mem, keep_stable_phase=False):
    # Apply the framework of repeating-squeezing-reordering
    # 1. repeating
    redundant_m = max(m, 2 * p)  # we add some redundant micro-batches to avoid unexpected bugs
    schedules = init_repeated_schedule(p, redundant_m, building_block)
    schedules = clear_invalid_index(schedules, redundant_m)
    init_peak_mem = get_peak_mem(schedules)
    if (m == redundant_m and init_peak_mem > max_mem) or init_peak_mem > 2 * p:
        return None, init_peak_mem, [6 * m] * p
    print_schedules(schedules, "after repeating")

    # 2. squeezing
    schedules = squeeze_without_change_order(schedules, redundant_m)
    print_schedules(schedules, "after squeezing")

    # 3. reordering
    # 3.a. reorder warm-up
    schedules = process_warmup_without_increasing_peak_mem(schedules, redundant_m)  # must work with m >= 2p
    schedules = squeeze_without_change_order(schedules, redundant_m)
    if keep_stable_phase:
        ending_index = [0] * p  # before second b
        for i in range(p):
            bb_cnt = 0
            for j in range(len(schedules[i])):
                if schedules[i][j] == 'b':
                    bb_cnt += 1
                    if bb_cnt >= 2:
                        ending_index[i] = j
                        break
        schedules = reorder_greedily_without_increasing_peak_mem(schedules, redundant_m, ending_index=ending_index)
    peak_mem = get_peak_mem(schedules)
    if debug:
        assert peak_mem <= init_peak_mem, f"{init_peak_mem}, {peak_mem}"
    if peak_mem > init_peak_mem:
        return None, init_peak_mem, [6 * m] * p

    if m < redundant_m:
        # 4. remove redundancy
        schedules = remove_redundancy(schedules, m)
        if m <= p and 2 * m <= max_mem:
            schedules = release_w_in_warmup_if_under_memory(schedules, peak_mem=min(2 * p, peak_mem))
        schedules = squeeze_without_change_order(schedules, m)
        print_schedules(schedules, "after removing redundancy")
        init_peak_mem = peak_mem = get_peak_mem(schedules)
        if peak_mem > max_mem:
            return None, peak_mem, [6 * m] * p

    # 3.b. reorder cool-down
    schedules = process_cooldown(schedules, m)
    if keep_stable_phase:
        starting_index = [0] * p
        for i in range(p):
            for j in range(len(schedules[i])):
                if schedules[i][j] == 'F':
                    starting_index[i] = j
        schedules = reorder_greedily_without_increasing_peak_mem(schedules, m, starting_index=starting_index)
    if not keep_stable_phase:
        reorder_greedily_without_increasing_peak_mem(schedules, m)
    schedules = relabel_w(schedules, m)
    print_schedules(schedules, "after reordering")
    peak_mem = get_peak_mem(schedules)
    if debug:
        assert peak_mem <= init_peak_mem, f"{init_peak_mem}, {peak_mem}"
    if peak_mem > init_peak_mem:
        return None, init_peak_mem, [6 * m] * p

    # return
    if not check_correctness(schedules, m, raise_exception=debug):
        return None, peak_mem, [6 * m] * p
    stage_bubbles = calc_bubble(schedules)
    if debug:
        print(peak_mem, stage_bubbles)
        print("-" * 100)
    return schedules, peak_mem, stage_bubbles


def fill_w_in_building_block(pattern):
    f, ff, b, bb, w, ww = 0, 1, 2, 3, 4, 5
    vis = [False] * pattern_size
    for v in pattern:
        if v >= 0:
            vis[v] = True
    assert pattern[b] >= 0 and pattern[bb] >= 0
    for v, vw in [(b, w), (bb, ww)]:
        for j in range(pattern_size):
            pos = (pattern[v] + j) % pattern_size
            if not vis[pos]:
                pattern[vw] = pos
                vis[pos] = True
                break
    return pattern


def get_building_block(pattern_0, offset_0, offset_1, len_0, p):
    # see Appendix A in the paper
    build_block = [pattern_0]
    for i in range(p - 1):
        last_pattern = build_block[i]
        new_pattern = [-1] * pattern_size
        vis = [False] * pattern_size
        if i < len_0:
            offset = offset_0
        else:
            offset = offset_1
        for v, v_o in enumerate(offset):
            pos = (last_pattern[v] + v_o + pattern_size) % pattern_size
            assert 0 <= pos < pattern_size
            if vis[pos]:
                return None
            vis[pos] = True
            new_pattern[v] = pos
        new_pattern = fill_w_in_building_block(new_pattern)
        build_block.append(new_pattern)
    return build_block


def schedule(p, m, cost, max_mem):
    f, ff, b, bb, w, ww = 0, 1, 2, 3, 4, 5
    available_starting_patterns = []
    # iterate available patterns for the first row/device of a building block
    for ff_i in range(1, pattern_size):
        for b_i in range(1, pattern_size):
            for bb_i in range(1, pattern_size):
                if ff_i == b_i or ff_i == bb_i or b_i == bb_i:
                    continue
                pattern = [0, ff_i, b_i, bb_i, -1, -1]
                pattern = fill_w_in_building_block(pattern)
                available_starting_patterns.append(pattern)

    # available uniform offsets, see Section 3.1 in the paper.
    available_offsets = [
        # [\delta_F^0, \delta_F^1, \delta_B^1, \delta_B^0]
        [1, -1, 1, -1],
        [2, -1, 2, -1],
        [3, -1, 3, -1],
        [4, -1, 4, -1],
        [5, -1, 5, -1]
    ]
    # available_starting_patterns = available_starting_patterns[:1]

    best_schedule = None
    best_bubble = None
    peak_mem2min_bubble = {}
    for pattern_0 in available_starting_patterns:
        for i_0 in range(len(available_offsets)):
            for i_1 in range(i_0 + 1):
                for len_0 in range(1, p):
                    offset_0 = available_offsets[i_0]
                    offset_1 = available_offsets[i_1]
                    build_block = get_building_block(pattern_0, offset_0, offset_1, len_0, p)
                    if build_block is None:
                        continue
                    s, peak_mem, bubbles = schedule_by_building_block(p, m, build_block, min(2 * p, max_mem))
                    if peak_mem > 2 * p or peak_mem > max_mem:
                        break
                    if s is None:
                        continue
                    max_bubble = evaluate_schedule(s, *cost)
                    if best_schedule is None or max_bubble < best_bubble:
                        best_schedule, best_bubble = s, max_bubble

                    max_bubble = max(bubbles)
                    min_bubble = min(peak_mem2min_bubble.get(peak_mem, max_bubble), max_bubble)
                    peak_mem2min_bubble[peak_mem] = min_bubble
    mem2bubble = {}
    for peak_mem in sorted(peak_mem2min_bubble.keys()):
        bubble = peak_mem2min_bubble[peak_mem]
        mem2bubble[peak_mem] = bubble
        # expected_bubble = max(0, 6 * p - 1 - 3 * peak_mem)
        expected_bubble = 3 * p - 1 - 3 * peak_mem + max(3 * p, p - 1 + (1 + (peak_mem + 1) // 2) * 2)
        # expected_bubble = 6 * p - 1 - 3 * peak_mem
        print(peak_mem, bubble, expected_bubble, "|", bubble - expected_bubble)
    print(mem2bubble)

    res = transform_schedule(best_schedule, *cost)
    return res
