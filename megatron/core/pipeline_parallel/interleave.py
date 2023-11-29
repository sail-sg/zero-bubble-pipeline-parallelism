

def get_interleave_schedule(_p, _n, _l, _f, _b, _w, _c):
    assert _n % _p == 0
    stage = [[] for _ in range(_p)]
    for rank in range(_p):
        warmup = (_p - rank - 1) * 2
        warmup += (_l - 1) * _p
        cooldown = rank * 2
        for _ in range(warmup):
            stage[rank].append(0)
        for i in range(_n * _l):
            if warmup + i < _n * _l:
                stage[rank].append(0)
            stage[rank].append(1)
        #     if i >= cooldown:
        #         stage[rank].append(2)
        # for _ in range(cooldown):
        #     stage[rank].append(2)
    fc = [0] * _p
    bc = [0] * _p
    for rank in range(_p):
        rank_str = " " * rank
        for i in range(_n * _l * 2):
            if stage[rank][i] == 0:
                if fc[rank] // _p % 2 == 0:
                    rank_str += 'F'
                else:
                    rank_str += 'f'
                fc[rank] += 1
            elif stage[rank][i] == 1:
                if bc[rank] // _p % 2 == 0:
                    rank_str += 'B'
                else:
                    rank_str += 'b'
                bc[rank] += 1
            else:
                rank_str += 'W'
        # print(rank_str)

    size = _p * _n * _l * 2

    def get_id(_i, _j, _k, _v):
        _kp, _kr = _k // _p, _k % _p
        _id = _kp * (_p * _l) + _v * _p + _kr
        return _i * _p * _n * _l + _j * _n * _l + _id

    t = [-1] * size
    e = [0] * _p
    fc = [0] * _p
    bc = [0] * _p
    for i in range(2 * _n * _l):
        ranks = []
        for rank in range(_p):
            if stage[rank][i] == 0:
                ranks.append(rank)
        for rank in range(_p - 1, -1, -1):
            if stage[rank][i] == 1:
                ranks.append(rank)
        # print(i, "->", ranks)
        for rank in ranks:
            if stage[rank][i] == 0:
                tmp = e[rank] + _f
                _id = fc[rank]
                _pk, v, _rk = _id // (_p * _l), (_id % (_p * _l)) // _p, _id % _p
                k = _pk * _p + _rk
                if rank > 0:
                    assert t[get_id(0, rank - 1, k, v)] > 0
                    tmp = max(tmp, t[get_id(0, rank - 1, k, v)] + _c + _f)
                elif _rk == 0 and v > 0:  # rank == 0
                    assert t[get_id(0, _p - 1, k, v - 1)] > 0
                    tmp = max(tmp, t[get_id(0, _p - 1, k, v - 1)] + _c + _f)
                e[rank] = tmp
                t[get_id(0, rank, k, v)] = tmp
                fc[rank] += 1
            elif stage[rank][i] == 1:
                tmp = e[rank] + _b
                _id = bc[rank]
                _pk, v, _rk = _id // (_p * _l), (_id % (_p * _l)) // _p, _id % _p
                k = _pk * _p + _rk
                if rank < _p - 1:
                    # print(rank, _pk, v, _rk, bc[rank])
                    assert t[get_id(1, rank + 1, k, v)] > 0, "{}: {}, {}, {}".format(rank, _pk, v, _rk)
                    tmp = max(tmp, t[get_id(1, rank + 1, k, v)] + _c + _b)
                elif _rk == 0 and v > 0:
                    assert t[get_id(1, 0, k, v - 1)] > 0
                    tmp = max(tmp, t[get_id(1, 0, k, v - 1)] + _c + _b)
                e[rank] = tmp
                t[get_id(1, rank, k, v)] = tmp
                bc[rank] += 1
            else:
                assert False
                _id = i - fc[rank] - bc[rank]
                _pk, v, _rk = _id // (_p * _l), (_id % (_p * _l)) // _p, _id % _p
                k = _pk * _p + _rk
                tmp = e[rank] + _w
                e[rank] = tmp
                t[get_id(2, rank, k, v)] = tmp
    max_time = 0
    for rank in range(_p):
        max_time = max(max_time, e[rank])
        # print(rank, "->", e[rank])
    # exit(0)
    return max_time

# get_interleave_schedule(4, 8, 2, 1, 2, 0, 0)
# exit(0)

settings = [
    # p,   n,    f,    b,    w,  c,    h,  a,  l
    ( 8,  24, 1859, 1806,  934, 36, 2304, 24, 24),
    ( 8,  32, 1855, 1807,  928, 31, 2304, 24, 24),
    ( 8,  64, 1854, 1809,  928, 32, 2304, 24, 24),
    ( 8,  24, 2965, 2941, 1982, 29, 4096, 32, 32),
    ( 8,  32, 2972, 2933, 1950, 29, 4096, 32, 32),
    ( 8,  64, 2994, 2968, 1937, 28, 4096, 32, 32),
    (16,  48, 1133, 1129,  815, 42, 5120, 40, 48),
    (16,  64, 1130, 1130,  813, 38, 5120, 40, 48),
    (16, 128, 1136, 1135,  816, 41, 5120, 40, 48),
    (32,  96, 1040, 1020,  773, 46, 6144, 48, 64),
    (32, 128, 1039, 1020,  769, 46, 6144, 48, 64),
    (32, 256, 1038, 1024,  770, 46, 6144, 48, 64),
]

s = 1024
for p, n, f, b, w, c, h, a, l in settings:
    expected_time = (f + b + w) * n
    ll = l // p
    interleave_cost = get_interleave_schedule(p, n, ll, f / ll, (b + w) / ll, 0, c)
    interleave_bubble = (interleave_cost - expected_time) / interleave_cost
    print("%2d %3d %6.4f" % (p, n, interleave_bubble))
