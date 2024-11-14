from dataclasses import dataclass
from enum import Enum
from typing import List


class PassType(Enum):
    F = "F"
    B = "B"
    W = "W"
    E = " "

@dataclass(eq=True, frozen=True)
class Pass:
    type: PassType
    chunk: int
    device: int
    micro: int

    def is_nan(self):
        if self.type == PassType.E or self.chunk == -1 or self.device == -1:
            return True
        return False

    def get_model_layer(self, device_num):
        if self.is_nan():
            return -1
        model_layer = self.chunk * device_num + self.device
        return model_layer

    def char(self):
        if self.is_nan():
            return " " * 4
        # if self.type == PassType.F and self.micro == 0:
        #     return " " * 7
        if self.chunk == 0:
            return "{}-{} ".format(self.type.value, self.micro)
        else:
            return "{}-{} ".format(self.type.value.lower(), self.micro)

Schedule = List[List[Pass]]
none_pass = Pass(type=PassType.E, chunk=-1, device=-1, micro=-1)

def get_zb_v_schedule(pp_size: int, num_micro: int) -> Schedule:
    zb_v_schedule = []
    n_micro = max(2 * pp_size - 1, num_micro)
    for rank in range(pp_size):
        schedule_this_rank = [none_pass for _ in range(rank)]
        f0_cnt, f1_cnt, b0_cnt, b1_cnt = 0, 0, 0, 0
        # warm-up phase
        warmup_n1 = 2 * (pp_size - rank) - 1
        for _ in range(warmup_n1):
            schedule_this_rank.append(Pass(type=PassType.F, chunk=0, device=rank, micro=f0_cnt))
            f0_cnt += 1
        warmup_n2 = rank
        for _ in range(warmup_n2):
            schedule_this_rank.append(Pass(type=PassType.F, chunk=1, device=rank, micro=f1_cnt))
            f1_cnt += 1
            schedule_this_rank.append(Pass(type=PassType.F, chunk=0, device=rank, micro=f0_cnt))
            f0_cnt += 1
        warmup_n3 = pp_size - rank
        for _ in range(warmup_n3):
            schedule_this_rank.append(Pass(type=PassType.F, chunk=1, device=rank, micro=f1_cnt))
            f1_cnt += 1
            schedule_this_rank.append(Pass(type=PassType.B, chunk=1, device=rank, micro=b1_cnt))
            schedule_this_rank.append(Pass(type=PassType.W, chunk=1, device=rank, micro=b1_cnt))
            b1_cnt += 1
        # stable phase
        while f1_cnt < f0_cnt or f0_cnt < n_micro:
            if f0_cnt < n_micro:
                schedule_this_rank.append(Pass(type=PassType.F, chunk=0, device=rank, micro=f0_cnt))
                f0_cnt += 1
            schedule_this_rank.append(Pass(type=PassType.B, chunk=0, device=rank, micro=b0_cnt))
            schedule_this_rank.append(Pass(type=PassType.W, chunk=0, device=rank, micro=b0_cnt))
            b0_cnt += 1

            schedule_this_rank.append(Pass(type=PassType.F, chunk=1, device=rank, micro=f1_cnt))
            f1_cnt += 1
            schedule_this_rank.append(Pass(type=PassType.B, chunk=1, device=rank, micro=b1_cnt))
            schedule_this_rank.append(Pass(type=PassType.W, chunk=1, device=rank, micro=b1_cnt))
            b1_cnt += 1
        # cool-down phase
        w0_cnt, w1_cnt = b0_cnt, b1_cnt
        cooldown_n1 = rank
        for _ in range(cooldown_n1):
            schedule_this_rank.append(Pass(type=PassType.B, chunk=0, device=rank, micro=b0_cnt))
            b0_cnt += 1
            schedule_this_rank.append(Pass(type=PassType.B, chunk=1, device=rank, micro=b1_cnt))
            b1_cnt += 1
        cooldown_n2 = pp_size - rank
        for _ in range(cooldown_n2):
            schedule_this_rank.append(Pass(type=PassType.B, chunk=0, device=rank, micro=b0_cnt))
            b0_cnt += 1
            schedule_this_rank.append(Pass(type=PassType.W, chunk=0, device=rank, micro=w0_cnt))
            w0_cnt += 1
        while w1_cnt < b1_cnt:
            schedule_this_rank.append(Pass(type=PassType.W, chunk=1, device=rank, micro=w1_cnt))
            w1_cnt += 1
        while w0_cnt < b0_cnt:
            schedule_this_rank.append(Pass(type=PassType.W, chunk=0, device=rank, micro=w0_cnt))
            w0_cnt += 1

        assert w0_cnt == b0_cnt and b0_cnt == f0_cnt
        assert w1_cnt == b1_cnt and b1_cnt == f1_cnt
        # remove redundant micros
        schedule_this_rank = [node if node.micro < num_micro else none_pass for node in schedule_this_rank]
        zb_v_schedule.append(schedule_this_rank)
    print_schedule(zb_v_schedule, debug=True)
    return zb_v_schedule


def print_schedule(schedule: Schedule, info: str = "", debug: bool = False):
    if not debug:
        return
    print(">" * 50, info)
    for d_i in range(len(schedule)):
        str_i = ""
        for node in schedule[d_i]:
            str_i += node.char()
        print(str_i)
    print(info, "<" * 50)

get_zb_v_schedule(4, 8)
