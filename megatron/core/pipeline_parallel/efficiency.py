from auto_schedule import do_heuristic_search as auto_schedule, GraphConfig
from v_schedule import PipelineGraph
import matplotlib.pyplot as plt
import numpy as np

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


def get_hand_schedule(_p, _n, _f, _b, _w, _c, warmup_c=2):
    assert _n >= 2 * _p
    stage = [[] for _ in range(_p)]
    for rank in range(_p):
        warmup = (_p - rank - 1) * warmup_c
        for _ in range(warmup):
            stage[rank].append(0)
        for i in range(_n):
            if warmup + i < _n:
                stage[rank].append(0)
            stage[rank].append(1)
            if warmup + i >= (_p - 1) * warmup_c:
                stage[rank].append(2)
        for _ in range((_p - 1) * warmup_c - warmup):
            stage[rank].append(2)
    labels = ["F", "B", "W"]
    for rank in range(_p):
        rank_str = " " * rank
        for i in range(_n * 3):
            rank_str += labels[stage[rank][i]]
        # print(rank_str)
    size = _p * _n * 3
    def get_id(_i, _j, _k):
        return _i * _p * _n + _j * _n + _k
    t = [-1] * size
    e = [0] * _p
    fc = [0] * _p
    bc = [0] * _p
    for i in range(3 * _n):
        for rank in range(_p):
            last = e[rank]
            if stage[rank][i] == 0:
                tmp = e[rank] + _f
                if rank > 0:
                    assert t[get_id(0, rank - 1, fc[rank])] > 0
                    tmp = max(tmp, t[get_id(0, rank - 1, fc[rank])] + _c + _f)
                e[rank] = tmp
                t[get_id(0, rank, fc[rank])] = tmp
                fc[rank] += 1
            elif stage[rank][i] == 1:
                tmp = e[rank] + _b
                if rank < _p - 1:
                    assert t[get_id(1, rank + 1, bc[rank])] > 0
                    tmp = max(tmp, t[get_id(1, rank + 1, bc[rank])] + _c + _b)
                e[rank] = tmp
                t[get_id(1, rank, bc[rank])] = tmp
                bc[rank] += 1
            else:
                tmp = e[rank] + _w
                e[rank] = tmp
                t[get_id(2, rank, i - fc[rank] - bc[rank])] = tmp
            # if rank == _p - 1:
            #     print(_f, _b, _w, _c, "->", rank, i, stage[rank][i], e[rank], e[rank] - last)
    max_time = 0
    for rank in range(_p):
        if warmup_c == 2:
            max_time = max(max_time, e[rank] - t[get_id(0, rank, 0)] + _f)
        else:
            max_time = max(max_time, e[rank])
        # print(rank, "->", e[rank])
    # exit(0)
    return max_time

# get_hand_schedule(4, 8, 1, 1, 1, 0, warmup_c=1)
# exit(0)


def iclr_bubble_rates():
    settings = [
        # p,   n,     f,     b,     w,   c,    h,  a,  l
        ( 8,  24, 18522, 18086,  9337, 601, 2304, 24, 24),
        ( 8,  32, 18513, 18086,  9331, 626, 2304, 24, 24),
        ( 8,  64, 18546, 18097,  9321, 762, 2304, 24, 24),
        ( 8,  24, 29718, 29444, 19927, 527, 4096, 32, 32),
        ( 8,  32, 29802, 29428, 19530, 577, 4096, 32, 32),
        ( 8,  64, 29935, 29621, 19388, 535, 4096, 32, 32),
        (16,  48, 11347, 11248,  8132, 377, 5120, 40, 48),
        (16,  64, 11307, 11254,  8101, 379, 5120, 40, 48),
        (16, 128, 11325, 11308,  8109, 378, 5120, 40, 48),
        (32,  96, 10419, 10207,  7715, 408, 6144, 48, 64),
        (32, 128, 10408, 10204,  7703, 408, 6144, 48, 64),
        (32, 256, 10402, 10248,  7698, 460, 6144, 48, 64),
    ]

    s = 1024
    for p, n, f, b, w, c, h, a,  l in settings:
        mem_f = 34 * h + 5 * a * s
        mem_w = - 32 * h
        mem_b = - mem_w - mem_f
        expected_time = (f + b + w) * n
        # zb-2p
        best_time, _, _ = auto_schedule(
            p,
            n,
            GraphConfig(
                mem_f=mem_f,
                mem_b=mem_b,
                mem_w=mem_w,
                max_mem=2 * p * mem_f,
                cost_f=f,
                cost_b=b,
                cost_w=w,
                cost_comm=c,
                print_scaling=1
            ),
        )
        p2_bubble = (best_time - expected_time) / best_time
        # zb-1p
        best_time, _, _ = auto_schedule(
            p,
            n,
            GraphConfig(
                mem_f=mem_f,
                mem_b=mem_b,
                mem_w=mem_w,
                max_mem=p * mem_f,
                cost_f=f,
                cost_b=b,
                cost_w=w,
                cost_comm=c,
                print_scaling=1
            ),
        )
        p1_bubble = (best_time - expected_time) / best_time
        # zb-h2
        h2_cost = get_hand_schedule(p, n, f, b, w, c, warmup_c=2)
        assert h2_cost - expected_time == (p - 1) * (f + b + 2 * c - 2 * w)
        h2_bubble = (h2_cost - expected_time) / h2_cost
        # zb-h1
        assert get_hand_schedule(p, n, f, b, w, 0, warmup_c=1) - expected_time == (p - 1) * (f + b - w)
        h1_cost = get_hand_schedule(p, n, f, b, w, c, warmup_c=1)
        h1_bubble = (h1_cost - expected_time) / h1_cost
        # 1F1B
        assert get_hand_schedule(p, n, f, b + w, 0, 0, warmup_c=1) - expected_time == (p - 1) * (f + b + w)
        cost_1f1b = get_hand_schedule(p, n, f, b + w, 0, c, warmup_c=1)
        bubble_1f1b = (cost_1f1b - expected_time) / cost_1f1b
        # interleaved 1F1B
        ll = l // p
        assert get_interleave_schedule(p, n, ll, f, (b + w), 0, 0) - expected_time * ll == (p - 1) * (f + b + w)
        interleave_cost = get_interleave_schedule(p, n, ll, f / ll, (b + w) / ll, 0, c)
        interleave_bubble = (interleave_cost - expected_time) / interleave_cost

        div = 1000
        print("%6.4f& %6.4f& %6.4f& %6.4f& %6.4f& \\textbf{%6.4f}" % (bubble_1f1b, interleave_bubble, h1_bubble, h2_bubble, p1_bubble, p2_bubble))
        # print("%3d& %6.3f& %6.3f& %6.3f& %5.3f" % (n, f / div, b / div, w / div, c / div))


def arxiv_bubble_rates():
    settings = [
        # p,   n,    f,     b,     w,   c,    h,  a,  l
        (16, 48, 14407, 14380,  9676, 1610, 4096, 32, 32),
        (16, 64, 14412, 14393,  9688, 1621, 4096, 32, 32),
        (16, 128,14316, 14306,  9639, 1619, 4096, 32, 32),
        (24, 72,  6763,  6969,  5251,  755, 5120, 40, 48),
        (24, 96,  6783,  6984,  5259,  758, 5120, 40, 48),
        (24, 192, 6785,  6990,  5260,  770, 5120, 40, 48),
        (32,  96, 9458,  9748,  7288,  879, 6144, 48, 64),
        (32, 128, 9469,  9744,  7306,  892, 6144, 48, 64),
        (32, 256, 9447,  9644,  7193,  887, 6144, 48, 64),
    ]

    s = 1024
    for p, n, _f, _b, _w, c, h, a, l in settings:
        f = _f * 2
        b = _b * 2
        w = _w * 2

        mem_f = 34 * h + 5 * a * s
        mem_w = - 32 * h
        mem_b = - mem_w - mem_f
        expected_time = (f + b + w) * n
        # zb-2p
        best_time, _, _ = auto_schedule(
            p,
            n,
            GraphConfig(
                mem_f=mem_f,
                mem_b=mem_b,
                mem_w=mem_w,
                max_mem=2 * p * mem_f,
                cost_f=f,
                cost_b=b,
                cost_w=w,
                cost_comm=c,
                print_scaling=1
            ),
        )
        p2_bubble = (best_time - expected_time) / best_time
        # zb-1p
        best_time, _, _ = auto_schedule(
            p,
            n,
            GraphConfig(
                mem_f=mem_f,
                mem_b=mem_b,
                mem_w=mem_w,
                max_mem=p * mem_f,
                cost_f=f,
                cost_b=b,
                cost_w=w,
                cost_comm=c,
                print_scaling=1
            ),
        )
        p1_bubble = (best_time - expected_time) / best_time
        # zb-h2
        h2_cost = get_hand_schedule(p, n, f, b, w, c, warmup_c=2)
        assert h2_cost - expected_time == (p - 1) * (f + b + 2 * c - 2 * w)
        h2_bubble = (h2_cost - expected_time) / h2_cost
        # zb-h1
        assert get_hand_schedule(p, n, f, b, w, 0, warmup_c=1) - expected_time == (p - 1) * (f + b - w)
        h1_cost = get_hand_schedule(p, n, f, b, w, c, warmup_c=1)
        h1_bubble = (h1_cost - expected_time) / h1_cost
        # 1F1B
        assert get_hand_schedule(p, n, f, b + w, 0, 0, warmup_c=1) - expected_time == (p - 1) * (f + b + w)
        cost_1f1b = get_hand_schedule(p, n, f, b + w, 0, c, warmup_c=1)
        bubble_1f1b = (cost_1f1b - expected_time) / cost_1f1b
        # interleaved 1F1B
        ll = l // p
        assert get_interleave_schedule(p, n, ll, f, (b + w), 0, 0) - expected_time * ll == (p - 1) * (f + b + w)
        interleave_cost = get_interleave_schedule(p, n, ll, f / ll, (b + w) / ll, 0, c)
        interleave_bubble = (interleave_cost - expected_time) / interleave_cost

        graph = PipelineGraph(
            n_stage=p,
            n_micro=n,
            f_cost=_f,
            b_cost=_b,
            w_cost=_w,
            c_cost=c,
            f_mem=mem_f,
            b_mem=mem_b,
            w_mem=mem_w,
        )
        zbv_cost = graph.get_v_schedule(only_run_time=True)
        zbv_bubble = (zbv_cost - expected_time) / zbv_cost

        div = 1000
        print("%6.4f& %6.4f& %6.4f& %6.4f& %6.4f" % (bubble_1f1b, interleave_bubble, h1_bubble, h2_bubble, zbv_bubble))
        # print("%3d& %6.3f& %6.3f& %6.3f& %5.3f" % (n, f / div, b / div, w / div, c / div))

# iclr_bubble_rates()
arxiv_bubble_rates()

# with open("mem2bubble.npy", "wb") as file:
#     np.save(file, xx)
#     np.save(file, yy)
#
# plt.plot(xx, yy)
# plt.ylabel("Bubble rate")
# plt.xlabel("$M_{limit}/M_B/p$")
# plt.gca().set_ylim(bottom=0)
# # plt.show()
# plt.savefig("mem2bubble.png")

"""
 8&  24& 18.522& 18.086&  9.337& 0.601& 0.2431& 0.1585& 0.1083& 0.1585& \textbf{0.0433} \\ \hline
 8&  32& 18.513& 18.086&  9.331& 0.626& 0.1985& 0.1242& 0.0837& 0.1242& \textbf{0.0039} \\ \hline
 8&  64& 18.546& 18.097&  9.321& 0.762& 0.1240& 0.0674& 0.0444& 0.0674& \textbf{0.0026} \\ \hline
 8&  24& 29.718& 29.444& 19.927& 0.527& 0.2347& 0.1323& 0.0698& 0.1323& \textbf{0.0029} \\ \hline
 8&  32& 29.802& 29.428& 19.530& 0.577& 0.1898& 0.1045& 0.0559& 0.1045& \textbf{0.0022} \\ \hline
 8&  64& 29.935& 29.621& 19.388& 0.535& 0.1091& 0.0554& 0.0294& 0.0554& \textbf{0.0010} \\ \hline
16&  48& 11.347& 11.248&  8.132& 0.377& 0.2552& 0.1397& 0.0672& 0.1397& \textbf{0.0066} \\ \hline
16&  64& 11.307& 11.254&  8.101& 0.379& 0.2082& 0.1088& 0.0516& 0.1088& \textbf{0.0054} \\ \hline
16& 128& 11.325& 11.308&  8.109& 0.378& 0.1251& 0.0576& 0.0266& 0.0576& \textbf{0.0028} \\ \hline
32&  96& 10.419& 10.207&  7.715& 0.408& 0.2646& 0.1421& 0.0641& 0.1421& \textbf{0.0038} \\ \hline
32& 128& 10.408& 10.204&  7.703& 0.408& 0.2168& 0.1106& 0.0490& 0.1106& \textbf{0.0029} \\ \hline
32& 256& 10.402& 10.248&  7.698& 0.460& 0.1352& 0.0594& 0.0257& 0.0594& \textbf{0.0018} \\ \hline

"""