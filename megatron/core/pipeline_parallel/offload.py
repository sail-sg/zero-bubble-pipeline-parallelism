import contextlib
import torch
from collections import defaultdict
from torch.autograd.graph import saved_tensors_hooks
from enum import Enum
import os
import re
import uuid

from megatron.core import parallel_state


def checksum(tensor):
    with torch.no_grad():
        if tensor.dtype == torch.half:
            return torch.mean(tensor * tensor).sum().item()
        else:
            return 0

def is_a_view(x, y):
    return x.storage().data_ptr() == y.storage().data_ptr() and x.storage_offset() == y.storage_offset() and x.numel() == y.numel()

def tensor_info(tensor):
    return (tensor.shape, tensor.layout, tensor.dtype, tensor.stride())

def save_rng_states():
    from megatron.core.tensor_parallel.random import get_cuda_rng_tracker
    return torch.get_rng_state(), torch.cuda.get_rng_state(), get_cuda_rng_tracker().get_states()

def restore_rng_states(states):
    from megatron.core.tensor_parallel.random import get_cuda_rng_tracker, _set_cuda_rng_state
    torch.set_rng_state(states[0])
    _set_cuda_rng_state(states[1])
    get_cuda_rng_tracker().set_states(states[2])


class PartialRecompute(saved_tensors_hooks):
    class RecomputeSaveType(Enum):
        PASS_THROUGH=1
        RECOMPUTE=2
    def _save_tensor(self, tensor):
        if self._next_recompute_tensor is not None and is_a_view(tensor, self._next_recompute_tensor[0]):
            packed = self._next_recompute_tensor[1:]
            self._next_recompute_tensor = None
            return PartialRecompute.RecomputeSaveType.RECOMPUTE, packed
        return PartialRecompute.RecomputeSaveType.PASS_THROUGH, tensor
    
    def _resume_tensor(self, packed):
        type, info = packed
        if type == PartialRecompute.RecomputeSaveType.RECOMPUTE:
            parents, function, rng_states = info
            with torch.no_grad():
                if rng_states is not None:
                    current_rng_states = save_rng_states()
                    restore_rng_states(rng_states)
                r = function(*parents)
                if rng_states is not None:
                    restore_rng_states(current_rng_states)
            return r
        return info

    def __init__(self):
        self._next_recompute_tensor = None
        super().__init__(self._save_tensor, self._resume_tensor)

    def _recompute_tensor(self, tensor, parents, function, rng_states=None):
        assert self._next_recompute_tensor is None
        self._next_recompute_tensor = (tensor, parents, function, rng_states)

partial_recompute = PartialRecompute()


class PairedBarrier:
    last_event = None
    event = None

    @classmethod
    def record(cls):
        # Only after the current exchange communication completes,
        # can we know the last event has been used by the peer device,
        # and we can safely free it.
        cls.last_event = cls.event
        from megatron.training import get_args
        if not get_args().paired_barrier:
            return
        cls.event = torch.cuda.Event(interprocess=True)
        cls.event.record()
        cls.ipc_handle = cls.event.ipc_handle()

    @classmethod
    def wait_peer(cls, peer: int = None):
        from megatron.training import get_args
        if not get_args().paired_barrier:
            return
        if peer is None:
            peer = torch.distributed.get_rank() ^ 1
            # Skip if peer is out of world size
            if peer >= torch.distributed.get_world_size():
                return

        if cls.event is None:
            # If no event is recorded, create a new one current state.
            cls.record()
        peer_handle = bytearray(len(cls.ipc_handle))

        s = torch.distributed.isend(tensor=torch.frombuffer(bytearray(cls.ipc_handle), dtype=torch.uint8), dst=peer)
        torch.distributed.recv(tensor=torch.frombuffer(peer_handle, dtype=torch.uint8), src=peer)
        s.wait()

        cls.last_event = None
        peer_event = torch.cuda.Event.from_ipc_handle(
          torch.cuda.current_device(), bytes(peer_handle))
        peer_event.wait()

class FakeActivationStore:
    @classmethod
    def barrier(cls):
        from megatron.training import get_args
        assert not get_args().offload_overlap_sr
        cls.offload()

    @classmethod
    def resume(cls):
        with torch.cuda.stream(get_offload_h2d_stream()):
            PairedBarrier.wait_peer()
        return 
    @classmethod
    def offload(cls):
        with torch.cuda.stream(get_offload_d2h_stream()):
            PairedBarrier.wait_peer()
        return

class ActivationStore(saved_tensors_hooks):
    @classmethod
    def recompute_tensor(cls, tensor, parents, function, rng_states=None):
        return partial_recompute._recompute_tensor(tensor, parents, function, rng_states)
            
    def __enter__(self):
        assert not hasattr(ActivationStore, '_current_activation_store') or ActivationStore._current_activation_store is None, "Nested offload not supported"
        ActivationStore._current_activation_store = self
        return super().__enter__()

    def __exit__(self, *args):
        super().__exit__(*args)
        ActivationStore._current_activation_store = None
    
    class State(Enum):
        NEW=0
        SAVING=1
        OFFLOADED=2
        OFFLOAD_RELEASED=3
        RESUME_PREPARED=4
        RESUMED=5
        RESUME_USED=6
        RESUME_RELEASED=7
    
    def _change_state(self, from_state, to_state):
        if isinstance(from_state, set):
            assert self._state in from_state
        else:
            assert self._state == from_state
        self._state = to_state
    
    class SaveType(Enum):
        OFFLOAD=1
        PASS_THROUGH=2
        RECOMPUTE=3
        ALIAS=4

    def _save_tensor(self, tensor):
        assert not self._offloaded
        self._change_state({ActivationStore.State.NEW, ActivationStore.State.SAVING}, ActivationStore.State.SAVING)
        if isinstance(tensor, torch.nn.parameter.Parameter):
            return ActivationStore.SaveType.PASS_THROUGH, tensor
        if tensor.numel() <= 1024:
            return ActivationStore.SaveType.PASS_THROUGH, tensor

        recompute = partial_recompute._save_tensor(tensor)
        if recompute[0] == PartialRecompute.RecomputeSaveType.RECOMPUTE:
            (parents, function, rng_states) = recompute[1]
            parent_handles = [self._save_tensor(x) for x in parents]
            return ActivationStore.SaveType.RECOMPUTE, (parent_handles, function, rng_states)
    
        for index, stored_tensor in enumerate(self._gpu_store):
            if is_a_view(tensor, stored_tensor):
                offset = tensor.storage_offset() - stored_tensor.storage_offset()
                stride = tensor.stride()
                shape = tensor.shape
                return ActivationStore.SaveType.ALIAS, (tensor.dtype, index, shape, stride, offset)

        self._gpu_store.append(tensor.data)
        if (len(self._offload_tensor_info) < len(self._gpu_store)):
            self._offload_tensor_info.append(tensor_info(tensor))
        else:
            assert(self._offload_tensor_info[len(self._gpu_store) - 1] == tensor_info(tensor))
        self._save_event.record()
        # print(f"rank {torch.distributed.get_rank()} Saving tensor id {len(self._gpu_store) - 1} {id(tensor)} {tensor.shape}, dtype {tensor.dtype}, device {tensor.device} storage {tensor.storage().data_ptr()}")
        return (ActivationStore.SaveType.OFFLOAD, len(self._gpu_store) - 1)
    
    def _resume_tensor(self, packed, remove_used=True):
        assert not self._offloaded
        type, info = packed
        self._change_state({ActivationStore.State.RESUMED, ActivationStore.State.RESUME_USED}, ActivationStore.State.RESUME_USED)
        if type == ActivationStore.SaveType.PASS_THROUGH:
            return info
        if type == ActivationStore.SaveType.RECOMPUTE:
            p_infos, function, rng_states = info
            parents = [self._resume_tensor(x, remove_used=False) for x in p_infos]
            return partial_recompute._resume_tensor((PartialRecompute.RecomputeSaveType.RECOMPUTE, (parents, function, rng_states)))
        if packed[0] == ActivationStore.SaveType.ALIAS:
            dtype, index, shape, stride, offset = packed[1]
            self._resume_event.wait()
            # print(f"rank {torch.distributed.get_rank()} Resuming alias tensor id {index} {shape}, offset {offset}")
            bin, o = self.index_offset[index]
            return torch.as_strided(self._continuous_gpu_buffer[dtype][bin], shape, stride, o + offset)
        assert type == ActivationStore.SaveType.OFFLOAD
        self._resume_event.wait()
        index = info
        ret = self._gpu_store[index]
        self._gpu_store[index] = None
        if remove_used:
            shape, layout, dtype, stride = self._offload_tensor_info[index]
            bin, offset = self.index_offset[index]
            all_freed = True
            for (i, (b, o)) in enumerate(self.index_offset):
                if self._gpu_store[i] is not None and b == bin and self._offload_tensor_info[i][2] == dtype:
                    all_freed = False
                    break
            if all_freed:
                self._continuous_gpu_buffer[dtype][bin] = None
        # print(f"rank {torch.distributed.get_rank()} Resuming tensor id {index} {ret.shape}, dtype {ret.dtype}, device {ret.device}")
        return ret

    def __init__(self, h2d_stream=None, d2h_stream=None):
        self._gpu_store=[]
        self._offloaded = False
        self._save_event = torch.cuda.Event()
        self._prepare_resume_event = torch.cuda.Event()
        self._resume_event = torch.cuda.Event()
        self._offload_complete_event = torch.cuda.Event()
        self._h2d_stream = h2d_stream
        self._d2h_stream = d2h_stream

        # Datastructures for offload
        self._continuous_cpu_buffer = None
        self._continuous_gpu_buffer = None
        self._offload_tensor_info = []
        self._index_offset = []
        self._index_cpu_buffer = []

        self._state = ActivationStore.State.NEW
        super().__init__(self._save_tensor, self._resume_tensor)
        
    def _allocate_cpu_buffers(self):
        if self._continuous_cpu_buffer is not None:
            return
        alignment=64
        
        
        def size_of_tensor(shape, stride):
            id_stride = list(sorted([(i, s) for i, s in enumerate(stride) if shape[i] != 1], key=lambda x: x[1]))
            size = 1
            for i, st in id_stride:
                assert size == st, f"stride {stride} size {shape} not continuous"
                size *= shape[i]
            return (size + (alignment - 1)) // alignment * alignment

        self.index_offset = []

        # dtype -> (size, id)
        type_tensors=defaultdict(list)

        for id, (shape, layout, dtype, stride) in enumerate(self._offload_tensor_info):
            assert layout == torch.strided
            # assert dtype == torch.half, f"Only half precision supported, got {dtype} shape {shape}"
            mysize = size_of_tensor(shape, stride)
            type_tensors[dtype].append((mysize, id))

        def nearest_power_of_2(x):
            return 2**(x-1).bit_length()

        def allocate_offset(tensors, max_split=4):
            total_size = sum([x[0] for x in tensors])
            aligned_size = nearest_power_of_2(total_size)
            bin_size = [aligned_size // 2**i for i in range(max_split)]
            bins = [0] * max_split
            tensors = sorted(tensors, key=lambda x: x[0], reverse=True)
            while True:
                bins[-1] += bin_size[-1]
                for i in range(max_split - 1, 0, -1):
                    if bins[i] > bin_size[i]:
                        bins[i] = 0
                        bins[i-1] += bin_size[i-1]

                solution_bins = [x for x in bins if x > 0]
                # print(solution_bins)
                if sum(solution_bins) < total_size:
                    continue
                current_bin = [0] * len(solution_bins)
                # id -> (bin, offset)
                solution = {}
                fit = True
                for size, id in tensors:
                    ok=False
                    for i in range(len(solution_bins)):
                        if current_bin[i] + size <= solution_bins[i]:
                            current_bin[i] += size
                            solution[id] = (i, current_bin[i] - size)
                            ok=True
                            break
                    if not ok:
                        fit = False
                        break
                if fit:
                    assert len(solution) == len(tensors)
                    assert all([x > 0 for x in current_bin])
                    return current_bin, solution
        
        import psutil
        print(f"rank {torch.distributed.get_rank()} before allocation rss {psutil.Process(os.getpid()).memory_info().rss / 1000000} MB")
        self._continuous_cpu_buffer = {}
        self.index_offset = [None] * len(self._offload_tensor_info)
        for (dtype, tensors) in type_tensors.items():
            from megatron.training import get_args
            if get_args().offload_continuous_buffers:
                bins, solution = allocate_offset(tensors, max_split=8)
            else:
                bins = [t[0] for t in tensors]
                solution = {tensors[i][1]: (i, 0) for i in range(len(tensors))}

            self._continuous_cpu_buffer[dtype] = [
                torch.empty([size], dtype=dtype, pin_memory=True, device='cpu') for size in bins]
            for id, (bin, offset) in solution.items():
                self.index_offset[id] = (bin, offset)
            print(f"rank {torch.distributed.get_rank()} after allocation {dtype} {bins} elements rss {psutil.Process(os.getpid()).memory_info().rss / 1000000} MB")

        # Print stats
        for dtype, tensors in type_tensors.items():
            total_size = sum([x[0] for x in tensors])
            allocated_size = sum([x.numel() for x in self._continuous_cpu_buffer[dtype]])
            aligned_size  = sum([nearest_power_of_2(x.numel()) for x in self._continuous_cpu_buffer[dtype]])
            print(f"rank {torch.distributed.get_rank()} Allocated {allocated_size / 1000000} M elements for {len(tensors)} tensors of type {dtype} total length {total_size} aligned size {aligned_size}")
        

        for index, (shape, layout, dtype, stride) in enumerate(self._offload_tensor_info):
            bin, offset = self.index_offset[index]
            ctensor = torch.as_strided(self._continuous_cpu_buffer[dtype][bin], shape, stride, offset)
            self._index_cpu_buffer.append(ctensor)

    @torch.no_grad()
    @torch.cuda.nvtx.range("Offload")
    def offload(self):
        self._change_state(ActivationStore.State.SAVING, ActivationStore.State.OFFLOADED)
        assert not self._offloaded
        
        size=0
        storage_size=0
        storages = set()
        
        with torch.cuda.stream(self._d2h_stream) if self._d2h_stream else contextlib.nullcontext():
            self._save_event.wait()
            self._allocate_cpu_buffers()
            PairedBarrier.wait_peer()
            for index, tensor in enumerate(self._gpu_store):
                buffer = self._index_cpu_buffer[index]
                assert buffer.shape == tensor.shape
                buffer.copy_(tensor, non_blocking=True)
                size+=tensor.numel()
                if tensor.storage().data_ptr() not in storages:
                    # print(f"rank {torch.distributed.get_rank()} Storage of tensor {tensor.shape} size {tensor.storage().size()/1000000} MB not in set")
                    storages.add(tensor.storage().data_ptr())
                    storage_size+=tensor.storage().nbytes()
                else:
                    # print(f"rank {torch.distributed.get_rank()} Storage of tensor {tensor.shape} size {tensor.storage().size()/1000000} MB already in set")
                    pass
                # print(f"Saving buffer to cpu shape {buffer.shape}, dtype {buffer.dtype}, device {buffer.device}")
            PairedBarrier.record()
            self._offload_complete_event.record()
        # print(f"rank {torch.distributed.get_rank()} Offloaded {size / 1000000000} Billion elements, {len(self._gpu_store)} tensors, storage size {storage_size / 1000000000} GBytes")
        
        self._offloaded = True

    @torch.no_grad()
    @torch.cuda.nvtx.range("OffloadRelease")
    def offload_release(self):
        self._change_state(ActivationStore.State.OFFLOADED, ActivationStore.State.OFFLOAD_RELEASED)
        assert self._offloaded
        if self._d2h_stream is not None:
            self._offload_complete_event.wait()
        self._gpu_store.clear()

    @torch.no_grad()
    @torch.cuda.nvtx.range("PrepareResume")
    def prepare_resume(self):
        self._change_state(ActivationStore.State.OFFLOAD_RELEASED, ActivationStore.State.RESUME_PREPARED)
        assert self._offloaded
        self._continuous_gpu_buffer = {
            dtype: [torch.empty_like(x, device='cuda') for x in bins] for dtype, bins in self._continuous_cpu_buffer.items()}
        for index, (shape, layout, dtype, stride) in enumerate(self._offload_tensor_info):
            bin, offset = self.index_offset[index]
            gtensor = torch.as_strided(self._continuous_gpu_buffer[dtype][bin], shape, stride, offset)
            self._gpu_store.append(gtensor)
        
        self._prepare_resume_event.record()


    @torch.no_grad()
    @torch.cuda.nvtx.range("Resume")
    def resume(self):
        self._change_state(ActivationStore.State.RESUME_PREPARED, ActivationStore.State.RESUMED)
        assert self._offloaded
        original_stream = torch.cuda.current_stream()
        with torch.cuda.stream(self._h2d_stream) if self._h2d_stream else contextlib.nullcontext():
            self._prepare_resume_event.wait()
            self._offload_complete_event.wait()
            PairedBarrier.wait_peer()
            for dtype, bins in self._continuous_cpu_buffer.items():
                for (cpu, gpu) in zip(bins, self._continuous_gpu_buffer[dtype]):
                    gpu.copy_(cpu, non_blocking=True)
            PairedBarrier.record()
            self._resume_event.record()
        self._offloaded = False

    @torch.no_grad()
    @torch.cuda.nvtx.range("ResumeRelease")
    def resume_release(self):
        self._change_state(ActivationStore.State.RESUME_USED, ActivationStore.State.RESUME_RELEASED)
        assert all([x is None for x in self._gpu_store])
        assert all([all([x is None for x in y]) for y in self._continuous_gpu_buffer.values()])
        self._resume_event.wait()
        
        self._gpu_store.clear()
        self._continuous_gpu_buffer.clear()

    def reset_state(self):
        self._change_state(ActivationStore.State.RESUME_RELEASED, ActivationStore.State.NEW)

offload_stream = None
d2h_stream = None
def get_offload_h2d_stream():
    global offload_stream
    if offload_stream is None:
        offload_stream = torch.cuda.Stream()
    return offload_stream
def get_offload_d2h_stream():
    from megatron.training import get_args
    if not get_args().offload_overlap_sr:
        return get_offload_h2d_stream()
    global d2h_stream
    if d2h_stream is None:
        d2h_stream = torch.cuda.Stream()
    return d2h_stream

# We expect the calling order for every store to be:
# get_for_offload
# offload
# offload_release
# prepare_resume
# resume
# resume_release
class ActivationStorePool:
    def __init__(self) -> None:
        self._pool = []
        self._stage_queues = [[] for x in range (6)]
    
    def get_for_offload(self) -> ActivationStore:
        if self._pool:
            ret = self._pool.pop(-1)
            ret.reset_state()
        else:
            ret = ActivationStore(get_offload_h2d_stream(), get_offload_d2h_stream())
        self._stage_queues[0].append(ret)
        return ret

    def pop_call_push(self, stage_idx, func):
        assert self._stage_queues[stage_idx]
        store = self._stage_queues[stage_idx].pop(0)
        ret = func(store)
        self._stage_queues[stage_idx + 1].append(store)
        return ret
    
    def offload(self):
        return self.pop_call_push(0, lambda x: x.offload())
    
    def offload_release(self):
        return self.pop_call_push(1, lambda x: x.offload_release())
    
    def prepare_resume(self):
        return self.pop_call_push(2, lambda x: x.prepare_resume())
    
    def resume(self):
        return self.pop_call_push(3, lambda x: x.resume())
    
    def resume_release(self, store_deprecated = None):
        self.pop_call_push(4, lambda x: x.resume_release())
        self._pool.append(self._stage_queues[5].pop(0))

    def is_empty(self):
        return sum([len(x) for x in self._stage_queues]) == 0
