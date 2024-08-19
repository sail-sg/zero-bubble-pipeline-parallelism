from megatron.core.timers import Timer


class ScheduleTimers:
    iter_counter = 0
    comm_time = 0
    concluded = False

    chunks = []

    def __init__(self):
        self.f = Timer('f')
        self.b = Timer('b')
        self.w = Timer('w')
        self.f_cnt = 0
        self.b_cnt = 0
        self.w_cnt = 0
        self.f_mem = 0
        self.b_mem = 0

    def conclusion(self):
        assert self.concluded
        assert self.f_cnt > 0
        assert self.b_cnt > 0
        avg_f = int(self.f.elapsed(reset=False) / self.f_cnt * 1000000)
        avg_b = int(self.b.elapsed(reset=False) / self.b_cnt * 1000000)
        avg_f_mem = self.f_mem / self.f_cnt // 1000000
        avg_b_mem = self.b_mem / self.b_cnt // 1000000
        if self.w_cnt > 0:
            avg_w = int(self.w.elapsed(reset=False) / self.w_cnt * 1000000)
        else:
            avg_w = avg_b
        avg_w_mem = 0 - avg_f_mem - avg_b_mem
        return (avg_f, avg_b, avg_w, int(self.comm_time * 1000000),
                avg_f_mem, avg_b_mem, avg_w_mem)

    @classmethod
    def for_chunk(cls, chunk):
        while len(cls.chunks) <= chunk:
            cls.chunks.append(cls())
        return cls.chunks[chunk]

    @classmethod
    def joint_conclusion(cls):
        ret = [x.conclusion() for x in cls.chunks]
        ret = list(zip(*ret))
        # C is shared bwteen chunks
        ret[3] = ret[3][0]
        return ret
