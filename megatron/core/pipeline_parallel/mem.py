from auto_schedule import do_heuristic_search as auto_schedule, GraphConfig
from v_schedule import PipelineGraph
import matplotlib.pyplot as plt
import numpy as np

# def get_rate(h, s, a):
#     mb = 34 * h + 5 * a * s
#     mw = 32 * h
#     return mb / mw

# config = [
#     (2304, 1024, 24),
#     (4096, 1024, 32),
#     (5120, 1024, 40),
#     (6144, 1024, 48),
# ]

# for i, j, k in config:
#     print(f"h={i}, s={j}, a={k}, mb/mw={get_rate(i, j, k)}")

def iclr_mem():
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
    xx, yy = [], []
    for p, n, f, b, w, c, h, a, _ in settings:
        mem_f = 34 * h + 5 * a * s
        mem_w = - 32 * h
        mem_b = - mem_w - mem_f
        expected_time = (f + b + w) * n
        x, y = [], []
        for k in range(0, 17):
            kk = 1 + k / 8
            best_time, _, _ = auto_schedule(
                p,
                n,
                GraphConfig(
                    mem_f=mem_f,
                    mem_b=mem_b,
                    mem_w=mem_w,
                    max_mem=int(kk * p * mem_f),
                    cost_f=f,
                    cost_b=b,
                    cost_w=w,
                    cost_comm=c,
                    print_scaling=1
                ),
            )
            bubble = (best_time - expected_time)
            x.append(kk)
            y.append(bubble / best_time)
            print(p, n, f, b, w, c, h, a, expected_time, bubble, best_time, bubble / best_time, k, sep=', ')
        xx.append(x)
        yy.append(y)

    xx = np.array(xx).transpose()
    yy = np.array(yy).transpose()

    with open("mem2bubble.npy", "wb") as file:
        np.save(file, xx)
        np.save(file, yy)

def arxiv_mem():
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
    xx, yy = [], []
    for p, n, _f, _b, _w, c, h, a, _ in settings:
        f = 2 * _f
        b = 2 * _b
        w = 2 * _w
        mem_f = 34 * h + 5 * a * s
        mem_w = - 32 * h
        mem_b = - mem_w - mem_f
        expected_time = (f + b + w) * n
        x, y = [], []
        for k in range(0, 17):
            kk = 1 + k / 8
            best_time, _, _ = auto_schedule(
                p,
                n,
                GraphConfig(
                    mem_f=mem_f,
                    mem_b=mem_b,
                    mem_w=mem_w,
                    max_mem=int(kk * p * mem_f),
                    cost_f=f,
                    cost_b=b,
                    cost_w=w,
                    cost_comm=c,
                    print_scaling=1
                ),
            )
            bubble = (best_time - expected_time)
            x.append(kk)
            y.append(bubble / best_time)
            print("zb", p, n, f, b, w, c, h, a, expected_time, bubble, best_time,
                  bubble / best_time, k, sep=', ')
        xx.append(x)
        yy.append(y)

    vx, vy = [], []
    for p, n, _f, _b, _w, c, h, a, _ in settings:
        f = 2 * _f
        b = 2 * _b
        w = 2 * _w
        mem_f = 34 * h + 5 * a * s
        mem_w = - 32 * h
        mem_b = - mem_w - mem_f
        expected_time = (f + b + w) * n
        x, y = [], []
        for k in range(0, 17):
            kk = 1 + k / 8
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
                max_mem=int(kk * p * mem_f * 2),
            )
            best_time = graph.get_v_schedule(only_run_time=True)
            bubble = (best_time - expected_time)
            x.append(kk)
            y.append(bubble / best_time)
            print("zbv", p, n, f, b, w, c, h, a, expected_time, bubble, best_time,
                  bubble / best_time, k, sep=', ')
        vx.append(x)
        vy.append(y)

    xx = np.array(xx).transpose()
    yy = np.array(yy).transpose()
    vx = np.array(vx).transpose()
    vy = np.array(vy).transpose()

    with open("mem2bubble.npy", "wb") as file:
        np.save(file, xx)
        np.save(file, yy)
        np.save(file, vx)
        np.save(file, vy)

arxiv_mem()

# plt.plot(xx, yy)
# plt.ylabel("Bubble rate")
# plt.xlabel("$M_{limit}/M_B/p$")
# plt.gca().set_ylim(bottom=0)
# # plt.show()
# plt.savefig("mem2bubble.png")
