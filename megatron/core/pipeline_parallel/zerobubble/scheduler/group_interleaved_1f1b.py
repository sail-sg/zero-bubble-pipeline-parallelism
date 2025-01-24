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
        save_activation: bool = True
        is_dependent: bool = True

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
            if not self.is_dependent:
                return model_layer
            if self.type == PassType.F:
                return model_layer - 1
            elif self.type == PassType.B:
                return model_layer + 1
            else:
                return model_layer

        def is_recompute(self):
            return self.type == PassType.F and self.save_activation and not self.is_dependent

        def char(self):
            if self.is_nan():
                return " " * 5
            # if self.type == PassType.F and self.micro == 0:
            #     return " " * 7
            if self.save_activation:
                if self.is_recompute():
                    return "{}{}-{} ".format("R", max(self.micro, 0), self.chunk)
                else:
                    return "{}{}-{} ".format(self.type.value, max(self.micro, 0), self.chunk)
            else:
                return "{}{}-{} ".format(self.type.value.lower(), max(self.micro, 0), self.chunk)

    Schedule = List[List[Pass]]
    none_pass = Pass(type=PassType.E, chunk=-1, device=-1, seq=-1, save_activation=False, is_dependent=False)

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
                    # f_index += min(d_i, extra_offset)
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

        unrolled_build_block = cls.unroll_building_block(building_block)
        # print(len(unrolled_build_block[0]), 6 * min_k * group_size * chunk_num - 3 * group_size * min_k + offset * (device_num - 1))
        return building_block, unrolled_build_block

    @classmethod
    def unroll_building_block(cls, building_block: Schedule):
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
    def repeat_building_block(cls, unrolled_build_block: Schedule, group_num: int, group_size: int = 1) -> Schedule:
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
        return repeated_schedule

    @classmethod
    def add_recomputation_pass(cls, repeated_schedule: Schedule, recompute_chunk_num: int) -> Schedule:
        schedule_with_recomputation = [
            [] for _i in range(len(repeated_schedule))
        ]
        for d_i in range(len(repeated_schedule)):
            for i_0, node in enumerate(repeated_schedule[d_i]):
                if node.type == PassType.B and node.chunk < recompute_chunk_num:
                    schedule_with_recomputation[d_i].append(
                        dataclasses.replace(node, type=PassType.F, save_activation=True, is_dependent=False, offset=-1)
                    )
                if node.type == PassType.F and node.chunk < recompute_chunk_num:
                    schedule_with_recomputation[d_i].append(
                        dataclasses.replace(node, save_activation=False, is_dependent=True, offset=-1)
                    )
                else:
                    schedule_with_recomputation[d_i].append(
                        dataclasses.replace(node, save_activation=True, is_dependent=True, offset=-1)
                    )
        return schedule_with_recomputation

    @classmethod
    def shift_schedule2meet_dependency(cls, schedule_to_shift: Schedule) -> Schedule:
        f_pass_indexes = [{} for _ in range(len(schedule_to_shift))]
        for i, schedule_i in enumerate(schedule_to_shift):
            for j, node in enumerate(schedule_i):
                if node.is_nan() or not node.is_dependent:
                    continue
                if node.type == PassType.F:
                    f_pass_indexes[i][(node.micro, node.chunk)] = j
        d = len(schedule_to_shift)
        extra_offset = d - 1
        for j, node in enumerate(schedule_to_shift[0]):
            if node.is_nan() or not node.is_dependent:
                continue
            if node.chunk > 0:
                offset = j - 1 - f_pass_indexes[d - 1][(node.micro, node.chunk - 1)]
                assert offset >= 0
                extra_offset = min(extra_offset, offset)
        # assert extra_offset < d
        shifted_schedule = [[] for _ in range(d)]
        for i, schedule_i in enumerate(schedule_to_shift):
            offset_i = min(i, extra_offset)
            for _ in range(offset_i):
                shifted_schedule[i].append(cls.none_pass)
            for node in schedule_i:
                shifted_schedule[i].append(dataclasses.replace(node))
            for _ in range(extra_offset - offset_i):
                shifted_schedule[i].append(cls.none_pass)
        return shifted_schedule

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
                    squeezed_len = max(squeezed_len, i + 1)
        for d_i in range(device_num):
            squeezed_schedule[d_i] = squeezed_schedule[d_i][:squeezed_len]
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
                if not node.save_activation:
                    continue
                if node.type == PassType.F:
                    mem += 1
                elif node.type == PassType.W:
                    mem -= 1
                peak_mem = max(peak_mem, mem)
            max_peak_mem = max(max_peak_mem, peak_mem)
        return max_peak_mem

    @classmethod
    def print_schedule(cls, schedule: Schedule, info: str = "", debug: bool = False):
        if not debug:
            return
        print(">" * 50, info)
        for d_i in range(len(schedule)):
            str_i = ""
            for node in schedule[d_i]:
                str_i += node.char()
            print(str_i)
        print(info, "<" * 50)


    def __init__(self, device_num: int, micro_num: int,
                 chunk_num: int = 2, min_group_size: int = 0, group_size: int = 1, recompute_chunk_num: int = 0,
                 debug: bool = False):
        if min_group_size == 0:
            min_group_size = (device_num + 1) // 2
        self.group_size = group_size
        self.device_num = device_num
        self.chunk_num = chunk_num
        self.build_block, self.unrolled_build_block = self.get_optimal_building_block(
            device_num, min_group_size=min_group_size, group_size=group_size, chunk_num=chunk_num)
        self.print_schedule(self.build_block, "building block", debug=debug)
        self.print_schedule(self.unrolled_build_block, "unrolled building block", debug=debug)
        group_num = (micro_num + group_size - 1) // group_size
        self.repeated_schedule = self.repeat_building_block(
            self.unrolled_build_block, group_num=group_num, group_size=group_size)
        self.repeated_schedule = self.remove_redundant_micro(self.repeated_schedule, micro_num)
        if group_size > 1:
            squeezed_schedule_without_recomputation = self.squeeze_without_change_order(self.repeated_schedule)
        else:
            squeezed_schedule_without_recomputation = self.repeated_schedule
        schedule_len_without_recomputation = len(squeezed_schedule_without_recomputation[0])
        self.print_schedule(self.repeated_schedule, "repeated schedule", debug=debug)
        peak_mem_before_recomputation = self.calculate_peak_memory(self.repeated_schedule)
        self.schedule_with_recomputation = self.add_recomputation_pass(self.repeated_schedule, recompute_chunk_num)
        peak_mem_after_recomputation = self.calculate_peak_memory(self.schedule_with_recomputation)
        self.print_schedule(self.schedule_with_recomputation, "add recomputation", debug=debug)
        self.shifted_schedule = self.shift_schedule2meet_dependency(self.schedule_with_recomputation)
        self.print_schedule(self.shifted_schedule, "shift for dependency", debug=debug)
        if group_size > 1:
            self.squeezed_schedule = self.squeeze_without_change_order(self.shifted_schedule)
        else:
            self.squeezed_schedule = self.shifted_schedule
        self.print_schedule(self.squeezed_schedule, "squeezed schedule", debug=debug)
        print(f"min_group_size: {min_group_size}, group_size: {group_size}, chunk: {chunk_num}, recompute_chunk: {recompute_chunk_num}")
        print(f"peak memory before->after recomputation: {peak_mem_before_recomputation} -> {peak_mem_after_recomputation}")
        print(f"schedule length before->after recomputation: {schedule_len_without_recomputation} -> {len(self.squeezed_schedule[0])}")
        print("-" * 50)

    def get_schedule(self) -> Schedule:
        return self.squeezed_schedule


def create_schedule(config, cpu_offload, recompute_granularity, recompute_method, recompute_num_layers, interleave_group_size, offload_chunk_num):
    from megatron.core.pipeline_parallel.zerobubble.scheduler.graph import GraphConfig
    assert isinstance(config, GraphConfig)

    max_fbw = max([tf + tb + tw for tf, tb, tw in zip(config.cost_f, config.cost_b, config.cost_w)])
    sum_f = sum(config.cost_f)
    sum_b = sum(config.cost_b)
    min_group_size = (config.n_stages + 1) // 2
    while max_fbw * min_group_size < max(sum_f, sum_b):
        min_group_size += 1
    assert min_group_size <= config.n_stages
    if recompute_granularity == "full":
        assert not cpu_offload
        assert recompute_method == "chunk"
        recompute_chunk_num = recompute_num_layers
        assert 1 <= recompute_chunk_num <= config.max_chunks
        group_size = 1
        best_group_size, min_schedule_len, min_peak_mem = -1, -1, -1
        best_group_schedule = None
        while group_size <= min_group_size:
            group_scheduler = GroupBuildingBlockScheduler(
                config.n_stages, config.n_micro, chunk_num=config.max_chunks,
                min_group_size=min_group_size, group_size=group_size, recompute_chunk_num=recompute_chunk_num)
            group_schedule = group_scheduler.get_schedule()
            schedule_len = len(group_schedule[0])
            peak_mem = GroupBuildingBlockScheduler.calculate_peak_memory(group_schedule)
            if best_group_size < 0 or (peak_mem, schedule_len) < (min_peak_mem, min_schedule_len):
                min_peak_mem = peak_mem
                min_schedule_len = schedule_len
                best_group_schedule = group_schedule
                best_group_size = group_size
            break
            group_size += 1
        assert best_group_schedule is not None
        print(f"best group size for recompute {best_group_size}, peak memory {min_peak_mem}, schedule length {min_schedule_len}")
    else:
        group_size, recompute_chunk_num = interleave_group_size, 0
        # if group_size < min_group_size:
        #     print(f"min interleave_group_size should be {min_group_size}, reset it from {group_size} to {min_group_size}")
        #     group_size = min_group_size
        if group_size > config.n_stages:
            print(f"max interleave_group_size should be {config.n_stages}, reset it from {group_size} to {config.n_stages}")
            group_size = config.n_stages

        group_scheduler = GroupBuildingBlockScheduler(
            config.n_stages, config.n_micro, chunk_num=config.max_chunks,
            min_group_size=min_group_size, group_size=group_size, recompute_chunk_num=recompute_chunk_num)
        best_group_schedule = group_scheduler.get_schedule()
        print("group size:", group_size)

    GroupBuildingBlockScheduler.print_schedule(best_group_schedule, "best schedule", debug=True)

    if recompute_chunk_num > 0:
        assert recompute_granularity == "full"

    if cpu_offload:
        offload_chunk_num = offload_chunk_num
        assert offload_chunk_num > 0
        assert recompute_granularity != "full"
    else:
        offload_chunk_num = 0
    return transform_schedule(config, best_group_schedule, offload_chunk_num, group_size==1)


def transform_schedule(config, best_group_schedule, offload_chunk_num, add_time=False):
    from megatron.core.pipeline_parallel.zerobubble.scheduler.graph import GraphConfig, ScheduledNode, FuncType
    local_order = []
    n_stages = len(best_group_schedule)
    func_types = {
        PassType.F: FuncType.F,
        PassType.B: FuncType.B,
        PassType.W: FuncType.W,
    }
    t4one_pass = max(config.cost_f)
    for i, schedule_i in enumerate(best_group_schedule):
        order = []
        for j, node in enumerate(schedule_i):
            if node.is_nan():
                continue
            assert node.device == i, f"{node.device}, {i}"
            ft = func_types[node.type]
            need_recompute = False
            if node.type == PassType.F:
                if node.is_recompute():
                    ft = FuncType.R
                need_recompute = not node.save_activation
            transformed_node = ScheduledNode(
                type=ft,
                stage=node.device,
                microbatch=node.micro,
                chunk=node.chunk,
                layer_group_idx=node.get_model_layer(n_stages),
                need_recompute=need_recompute,
                should_offload=node.chunk < offload_chunk_num,
            )
            if add_time:
                transformed_node = dataclasses.replace(
                    transformed_node,
                    start_time=j * t4one_pass,
                    completion_time=(j+1) * t4one_pass,
                )
            order.append(transformed_node)
        local_order.append(order)
    return local_order


def look_schedule():
    scheduler = GroupBuildingBlockScheduler(8, 32, chunk_num=3, min_group_size=4, group_size=1, recompute_chunk_num=0, debug=True)

# look_schedule()
