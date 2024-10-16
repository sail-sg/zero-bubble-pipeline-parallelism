import dataclasses
import math
from collections import Counter, deque
from dataclasses import dataclass
from enum import Enum
from typing import List


class PassType(Enum):
    F = "F"
    B = "B"
    W = "W"
    E = " "


class GroupBuildingBlockScheduler(object):

    @dataclass(eq=True, frozen=True)
    class Pass:
        type: PassType
        chunk: int
        device: int
        seq: int = 0
        micro: int = -1
        offset: int = -1

        def is_nan(self):
            if self.type == PassType.E or self.chunk == -1 or self.device == -1 or self.seq == -1:
                return True
            return False

        def get_model_layer(self, device_num):
            if self.is_nan():
                return -1
            model_layer = self.chunk * device_num + self.device
            return model_layer

        def get_dependent_model_layer(self, device_num):
            if self.is_nan():
                return -1
            model_layer = self.get_model_layer(device_num)
            if self.type == PassType.F:
                return model_layer - 1
            elif self.type == PassType.B:
                return model_layer + 1
            else:
                return model_layer

        def char(self):
            if self.is_nan():
                return " " * 5
            # if self.type == PassType.F and self.micro == 0:
            #     return " " * 7
            return "{}{}-{} ".format(self.type.value, max(self.micro, 0), self.chunk)

    Schedule = List[List[Pass]]
    none_pass = Pass(type=PassType.E, chunk=-1, device=-1, seq=-1)

    @classmethod
    def get_optimal_building_block(cls, device_num: int, min_group_size: int = 1, group_size: int = 1, chunk_num: int = 2):
        # assert math.gcd(chunk_num, device_num) == 1

        # f_offset, b_offset = 3, 3
        f_offset, b_offset = 1, 2
        offset = f_offset + b_offset
        min_k = (min_group_size + group_size - 1) // group_size
        gcd_kv = math.gcd(chunk_num, min_k)
        extra_offset = max(0, b_offset * device_num - 3 * min_k * group_size)
        assert extra_offset < device_num
        print(min_k, gcd_kv, group_size, min_group_size, chunk_num)

        building_block: cls.Schedule
        building_block = [
            [cls.none_pass for _ in range(3 * group_size * chunk_num)] for _i in range(device_num)
        ]
        # bb_len = 6 * group_size * chunk_num - 3 * group_size + offset * (device_num - 1)
        bb_len = 3 * group_size * chunk_num
        last_f_before_b = {}
        for c_i in range(chunk_num):
            for g_i in range(group_size):
                for d_i in range(device_num):
                    f_index = 3 * min_k * group_size * c_i + 3 * g_i + d_i * f_offset
                    f_index += 3 * group_size * (c_i // (chunk_num // gcd_kv))
                    f_index += min(d_i, extra_offset)
                    # f_index += max(d_i - (device_num - 1 - extra_offset), 0)
                    assert building_block[d_i][f_index % bb_len].is_nan()
                    building_block[d_i][f_index % bb_len] = cls.Pass(
                        type=PassType.F,
                        chunk=c_i,
                        device=d_i,
                        micro=g_i,
                        offset=f_index
                    )
                    if c_i == chunk_num - 1 and g_i == 0:
                        last_f_before_b[d_i] = f_index
        for c_i in range(chunk_num):
            for g_i in range(group_size):
                for d_i in range(device_num):
                    last_f_index = last_f_before_b[d_i]
                    first_b_index = last_f_index + (device_num - 1 - d_i) * f_offset + 1 + (device_num - 1 - d_i) * b_offset
                    b_index = first_b_index + 3 * min_k * group_size * c_i + 3 * g_i
                    b_index += 3 * group_size * (c_i // (chunk_num // gcd_kv))
                    assert building_block[d_i][b_index % bb_len].is_nan()
                    building_block[d_i][b_index % bb_len] = cls.Pass(
                        type=PassType.B,
                        chunk=chunk_num - 1 - c_i,
                        device=d_i,
                        micro=g_i,
                        offset=b_index
                    )
                    w_index = b_index + 1
                    assert building_block[d_i][w_index % bb_len].is_nan()
                    building_block[d_i][w_index % bb_len] = cls.Pass(
                        type=PassType.W,
                        chunk=chunk_num - 1 - c_i,
                        device=d_i,
                        micro=g_i,
                        offset=w_index
                    )

        unrolled_build_block = cls.unroll_build_block(building_block)
        # print(len(unrolled_build_block[0]), 6 * min_k * group_size * chunk_num - 3 * group_size * min_k + offset * (device_num - 1))
        cls.print_schedule(building_block)
        cls.print_schedule(unrolled_build_block)
        return building_block, unrolled_build_block

    @classmethod
    def unroll_build_block(cls, building_block: Schedule):
        device_num = len(building_block)
        bb_len = len(building_block[0])
        max_offset = 0
        for d_i in range(device_num):
            for node in building_block[d_i]:
                max_offset = max(max_offset, node.offset)
        unrolled_build_block = [
            [cls.none_pass for _ in range(max_offset + 1)] for _i in range(device_num)
        ]
        for d_i in range(device_num):
            count = 0
            for i in range(len(unrolled_build_block[d_i])):
                if building_block[d_i][i % bb_len].offset == i:
                    count += 1
                    unrolled_build_block[d_i][i] = building_block[d_i][i % bb_len]
                if count >= bb_len:
                    break
            # assert count == bb_len, f"{count}, {bb_len}"
        return unrolled_build_block

    @classmethod
    def repeat_building_block(cls, unrolled_build_block: Schedule, group_num: int, group_size: int = 1):
        bb_len = 0
        for node in unrolled_build_block[0]:
            if not node.is_nan():
                bb_len += 1
        max_len = len(unrolled_build_block[0]) + bb_len * (group_num - 1)
        repeated_schedule = [
            [cls.none_pass for _ in range(max_len)] for _i in range(len(unrolled_build_block))
        ]
        for d_i in range(len(unrolled_build_block)):
            for i_0, node in enumerate(unrolled_build_block[d_i]):
                if node.is_nan():
                    continue
                for m_i in range(group_num):
                    index = i_0 + bb_len * m_i
                    repeated_schedule[d_i][index] = cls.Pass(
                        type=node.type,
                        chunk=node.chunk,
                        device=node.device,
                        seq=node.seq,
                        micro=node.micro + m_i * group_size,
                        offset=node.offset
                    )
        cls.print_schedule(repeated_schedule)
        return repeated_schedule

    @classmethod
    def squeeze_without_change_order(cls, schedule: Schedule):
        device_num = len(schedule)
        chunk_num = 0
        for node in schedule[0]:
            chunk_num = max(chunk_num, node.chunk + 1)

        squeezed_schedule = [[] for _ in range(device_num)]
        finalized_keys = set()
        cur_index = [0] * device_num
        squeezed_len = 0
        max_len = len(schedule[0])
        for i in range(max_len):
            for d_i in range(device_num):
                while cur_index[d_i] < max_len and schedule[d_i][cur_index[d_i]].is_nan():
                    cur_index[d_i] += 1
                if cur_index[d_i] >= max_len:
                    squeezed_schedule[d_i].append(cls.none_pass)
                    continue
                node = schedule[d_i][cur_index[d_i]]
                model_layer = node.get_model_layer(device_num)
                prev_model_layer = node.get_dependent_model_layer(device_num)
                prev_model_layer = min(max(prev_model_layer, 0), chunk_num * device_num - 1)
                prev_key = (node.micro, node.type, node.seq, prev_model_layer)
                if model_layer == prev_model_layer or prev_key in finalized_keys:
                    squeezed_schedule[d_i].append(dataclasses.replace(node))
                    cur_index[d_i] += 1
                else:
                    squeezed_schedule[d_i].append(cls.none_pass)
            for d_i in range(device_num):
                node = squeezed_schedule[d_i][i]
                if not node.is_nan():
                    model_layer = node.get_model_layer(device_num)
                    node_key = (node.micro, node.type, node.seq, model_layer)
                    finalized_keys.add(node_key)
                    squeezed_len = max(squeezed_len, i)
        for d_i in range(device_num):
            squeezed_schedule = squeezed_schedule[:squeezed_len]
        cls.print_schedule(squeezed_schedule)
        return squeezed_schedule

    @classmethod
    def remove_redundant_micro(cls, schedule: Schedule, micro_num):
        for schedule_i in schedule:
            for idx, node in enumerate(schedule_i):
                if node.micro >= micro_num:
                    schedule_i[idx] = cls.none_pass
        return schedule

    @classmethod
    def calculate_peak_memory(cls, schedule: Schedule):
        max_peak_mem = 0
        for schedule_i in schedule:
            peak_mem, mem = 0, 0
            for node in schedule_i:
                if node.type == PassType.F:
                    mem += 1
                elif node.type == PassType.W:
                    mem -= 1
                peak_mem = max(peak_mem, mem)
            max_peak_mem = max(max_peak_mem, peak_mem)
        return max_peak_mem

    @classmethod
    def print_schedule(cls, schedule: Schedule):
        print(">" * 50)
        for d_i in range(len(schedule)):
            str_i = ""
            for node in schedule[d_i]:
                str_i += node.char()
            print(str_i)
        print("<" * 50)


    def __init__(self, device_num: int, micro_num: int, chunk_num: int = 2, min_group_size: int = 1, group_size: int = 1):
        """
        :param device_num:
        :param micro_num
        :param chunk_num:
        :param min_group_size: the minimal value of g satisfying g*(t_f+t_b+t_w) >= device_num * [max(t_f, t_b) + t_comm]
        :param group_size:
        """
        self.min_group_size = min_group_size
        self.group_size = group_size
        self.device_num = device_num
        self.chunk_num = chunk_num
        self.build_block, self.unrolled_build_block = self.get_optimal_building_block(
            device_num, min_group_size=min_group_size, group_size=group_size, chunk_num=chunk_num)
        group_num = (micro_num + group_size - 1) // group_size * group_size
        self.repeated_schedule = self.repeat_building_block(
            self.unrolled_build_block, group_num=group_num, group_size=group_size)
        self.repeated_schedule = self.remove_redundant_micro(self.repeated_schedule, micro_num)
        self.squeezed_schedule = self.squeeze_without_change_order(self.repeated_schedule)
        
    def get_schedule(self) -> Schedule:
        return self.squeezed_schedule


def create_schedule(config):
    from megatron.core.pipeline_parallel.zerobubble.scheduler.graph import GraphConfig, ScheduledNode, FuncType
    assert isinstance(config, GraphConfig)

    min_group_size = (config.n_stages + 1) // 2

    group_scheduler = GroupBuildingBlockScheduler(
        config.n_stages, config.n_micro, chunk_num=config.max_chunks,
        min_group_size=min_group_size, group_size=2)
    group_schedule = group_scheduler.get_schedule()

    local_order = []
    func_types = {
        PassType.F: FuncType.F,
        PassType.B: FuncType.B,
        PassType.W: FuncType.W,
    }
    for i, schedule_i in enumerate(group_schedule):
        order = []
        for node in schedule_i:
            if node.is_nan():
                continue
            assert node.device == i, f"{node.device}, {i}"
            order.append(ScheduledNode(
                type=func_types[node.type],
                stage=node.device,
                microbatch=node.micro,
                chunk=node.chunk,
                layer_group_idx=node.get_model_layer(config.n_stages)
            ))
        local_order.append(order)
    return local_order


# scheduler = GroupBuildingBlockScheduler(8, 8, chunk_num=3, min_group_size=4, group_size=2)
