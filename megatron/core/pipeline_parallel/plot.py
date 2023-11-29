import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import numpy as np

def iclr_plot():
    with open("/Users/qiph/Downloads/mem2bubble.npy", "rb") as file:
        xx = np.load(file)
        yy = np.load(file)

    print(xx.shape, yy.shape)

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

    model_name = ["1.5B Model, $p=8$", "6.2B Model, $p=8$", "14.6B Model, $p=16$", "28.3B Model, $p=32$"]

    font = {
        # 'family' : 'normal',
        # 'weight' : 'bold',
        'size'   : 15
    }

    matplotlib.rc('font', **font)

    fig, axs = plt.subplots(1, 4, sharex=True, sharey=True, figsize=(17, 3.6))
    for model_id in range(4):
        index = slice(model_id * 3, model_id * 3 + 3)
        for i in range(3):
            j = model_id * 3 + i
            axs[model_id].plot(xx[:, j], yy[:, j], label=f"#$Microbatches={settings[j][1]}$")
        axs[model_id].set_title(model_name[model_id])
        axs[model_id].legend()
    for ax in axs.flat:
        ax.set(xlabel="$M_{limit}$", ylabel="Bubble rate")
    # Hide x labels and tick labels for top plots and y ticks for right plots.
    for ax in axs.flat:
        ax.label_outer()
    gca = plt.gca()
    gca.xaxis.set_major_formatter(FormatStrFormatter("%.1f$pM_B$"))
    gca.set_ylim(bottom=0, top=0.18)
    # print(gca.axes.get_yticks(), gca.axes.get_yticklabels())
    gca.axes.set_xticks([1, 2, 3])
    # gca.axes.set_yticks([0.0, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.14, 0.16])
    # gca.axes.set_yticklabels([0.0, "", 0.04, "", 0.08, "", 0.12, "", 0.16])
    plt.subplots_adjust(top=0.90, bottom=0.18, left=0.06, right=0.98, hspace=0.40, wspace=0.14)
    # plt.rcParams.update({'font.size': 24})

    # fig = plt.figure(figsize=(13, 3.2))
    # gs = fig.add_gridspec(1, 4, wspace=0.05)
    # axs = gs.subplots(sharex=True, sharey=True)
    # for model_id in range(4):
    #     # plt.subplot(141 + model_id)
    #     index = slice(model_id * 3, model_id * 3 + 3)
    #     # plt.plot(xx[:, index], yy[:, index], label=np.array(["$n=3p$", "$n=4p$", "$n=8p$"]))
    #     for i in range(3):
    #         j = model_id * 3 + i
    #         axs[model_id].plot(xx[:, j], yy[:, j], label=f"$n={settings[j][1]}$")
    #     # plt.xlabel("$M_{limit}/M_B/p$")
    #     # gca = plt.gca()
    #     # gca.set_ylim(bottom=0, top=0.18)
    #     # gca.axes.get_yaxis().set_visible(False)
    #     # if model_id > 0:
    #     #     gca.axes.set_yticklabels([])
    #     if model_id == 0:
    #         axs[model_id].set_ylabel("Bubble rate")
    #     axs[model_id].set_title(model_name[model_id])
    #     axs[model_id].legend()


    # for model_id in range(4):
    #     plt.subplot(141 + model_id)
    #     index = slice(model_id * 3, model_id * 3 + 3)
    #     # plt.plot(xx[:, index], yy[:, index], label=np.array(["$n=3p$", "$n=4p$", "$n=8p$"]))
    #     for i in range(3):
    #         j = model_id * 3 + i
    #         plt.plot(xx[:, j], yy[:, j], label=f"$n={settings[j][1]}$")
    #     if model_id == 0:
    #         plt.ylabel("Bubble rate")
    #     # plt.xlabel("$M_{limit}/M_B/p$")
    #     gca = plt.gca()
    #     gca.set_ylim(bottom=0, top=0.18)
    #     # gca.axes.get_yaxis().set_visible(False)
    #     # if model_id > 0:
    #     #     gca.axes.set_yticklabels([])
    #     plt.title(model_name[model_id])
    #     plt.legend()
    #
    # plt.subplots_adjust(top=0.90, bottom=0.10, left=0.08, right=0.98, hspace=0.40, wspace=0.25)

    # plt.show()
    plt.savefig("Zero Bubble - Mem2bubble.pdf")


def arxiv_plot():
    with open("mem2bubble.npy", "rb") as file:
        xx = np.load(file)
        yy = np.load(file)
        vx = np.load(file)
        vy = np.load(file)

    print(xx.shape, yy.shape, vx.shape, vy.shape)

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

    model_name = []
    for p, n, _f, _b, _w, c, h, a, _ in settings:
        if p == 16:
            model = "6.2B"
        elif p == 24:
            model = "14.6B"
        else:
            model = "28.3B"
        model_name.append("{}, $p={}$, $m={}$".format(model, p, n))

    font = {
        # 'family' : 'normal',
        # 'weight' : 'bold',
        'size'   : 8
    }

    matplotlib.rc('font', **font)

    fig, axs = plt.subplots(3, 3, sharex=True, sharey=True)
    for model_id in range(9):
        i = model_id // 3
        j = model_id % 3
        axs[i][j].plot(xx[:, j * 3 + i], yy[:, j * 3 + i], label=f"ZB")
        axs[i][j].plot(vx[:, j * 3 + i], vy[:, j * 3 + i], label=f"ZB-V")
        axs[i][j].set_title(model_name[j * 3 + i])
        axs[i][j].legend()
    for ax in axs.flat:
        ax.set(xlabel="$M_{limit}$", ylabel="Bubble rate")
    # Hide x labels and tick labels for top plots and y ticks for right plots.
    for ax in axs.flat:
        ax.label_outer()
    gca = plt.gca()
    gca.xaxis.set_major_formatter(FormatStrFormatter("%.1f$pM_B$"))
    gca.set_ylim(bottom=0, top=0.18)
    # print(gca.axes.get_yticks(), gca.axes.get_yticklabels())
    gca.axes.set_xticks([1, 2, 3])
    # gca.axes.set_yticks([0.0, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.14, 0.16])
    # gca.axes.set_yticklabels([0.0, "", 0.04, "", 0.08, "", 0.12, "", 0.16])
    plt.subplots_adjust(top=0.95, bottom=0.10, left=0.10, right=0.95, hspace=0.40, wspace=0.16)
    # plt.rcParams.update({'font.size': 24})
    plt.savefig("Zero Bubble - Mem2bubble_ZBV.pdf")

arxiv_plot()
