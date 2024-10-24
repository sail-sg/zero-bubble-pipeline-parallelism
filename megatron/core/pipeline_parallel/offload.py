import contextlib
import torch
from collections import defaultdict
from torch.autograd.graph import saved_tensors_hooks
from enum import Enum
import os
import re
import uuid

def checksum(tensor):
    with torch.no_grad():
        if tensor.dtype == torch.half:
            return torch.mean(tensor * tensor).sum().item()
        else:
            return 0

def is_a_view(x, y):
    return x.storage().data_ptr() == y.storage().data_ptr() and x.storage_offset() == y.storage_offset() and x.numel() == y.numel()

def save_rng_states():
    from megatron.core.tensor_parallel.random import get_cuda_rng_tracker
    return torch.get_rng_state(), torch.cuda.get_rng_state(), get_cuda_rng_tracker().get_states()

def restore_rng_states(states):
    from megatron.core.tensor_parallel.random import get_cuda_rng_tracker, _set_cuda_rng_state
    torch.set_rng_state(states[0])
    _set_cuda_rng_state(states[1])
    get_cuda_rng_tracker().set_states(states[2])

class NumaManager:
    set = False
    @classmethod
    def set_affinity(cls):
        # Set affinity according to nvidia-smi result and rank
        if cls.set:
            return
        output = os.popen('nvidia-smi topo -m').read()
        local_rank = torch.distributed.get_rank() % torch.cuda.device_count()
        if os.environ.get('CUDA_VISIBLE_DEVICES') is not None:
            myrank = int(os.environ.get('CUDA_VISIBLE_DEVICES').split(',')[local_rank])
        else:
            myrank = torch.distributed.get_rank()

        for line in output.split('\n'):
            if line.startswith(f'GPU{myrank}\t'):
                # using regex to match pattern like 36-47
                result = re.search(r'(\d+)-(\d+)', line).groups()
                start = int(result[0])
                end = int(result[1])
                affinity = range(start, end + 1)
                os.sched_setaffinity(0, affinity)
                print(f"rank {torch.distributed.get_rank()} Setting affinity to {affinity}")
                cls.set = True
                break

class PartialRecompute(saved_tensors_hooks):
    def __init__(self):
        self._next_recompute_tensor = None
        self._recompute_store = {}
        class RecomputeSaveType(Enum):
            PASS_THROUGH=1
            RECOMPUTE=2

        def save_tensor(tensor):
            if self._next_recompute_tensor is not None and is_a_view(tensor, self._next_recompute_tensor[0]):
                id=uuid.uuid4()
                self._recompute_store[id] = self._next_recompute_tensor[1:]
                self._next_recompute_tensor = None
                return RecomputeSaveType.RECOMPUTE, id
            
            return RecomputeSaveType.PASS_THROUGH, tensor
        
        def resume_tensor(packed):
            type, id = packed
            if type == RecomputeSaveType.RECOMPUTE:
                parents, function, rng_states = self._recompute_store[id]
                self._recompute_store.pop(id)
                with torch.no_grad():
                    if rng_states is not None:
                        current_rng_states = save_rng_states()
                        restore_rng_states(rng_states)
                    r = function(*parents)
                    if rng_states is not None:
                        restore_rng_states(current_rng_states)
                    # print(f'rank {torch.distributed.get_rank()} recomputed tensor {r.shape} {checksum(r)} inputs {[checksum(x) for x in parents]}')
                # print(f'rank {torch.distributed.get_rank()} recompute registered for tensor {r.shape}')
                return r
            return id
        super().__init__(save_tensor, resume_tensor)

    def _recompute_tensor(self, tensor, parents, function, rng_states=None):
        assert self._next_recompute_tensor is None
        self._next_recompute_tensor = (tensor, parents, function, rng_states)

partial_recompute = PartialRecompute()

class ActivationStore(saved_tensors_hooks):
    @classmethod
    def recompute_tensor(cls, tensor, parents, function, rng_states=None):
        if hasattr(cls, '_current_activation_store') and cls._current_activation_store is not None:
            return cls._current_activation_store._recompute_tensor(tensor, parents, function, rng_states)
        else:
            return partial_recompute._recompute_tensor(tensor, parents, function, rng_states)
            
    def __enter__(self):
        assert not hasattr(ActivationStore, '_current_activation_store') or ActivationStore._current_activation_store is None, "Nested offload not supported"
        ActivationStore._current_activation_store = self
        return super().__enter__()

    def __exit__(self, *args):
        super().__exit__(*args)
        ActivationStore._current_activation_store = None

    def __init__(self, h2d_stream=None, d2h_stream=None):
        # # TODO: Read this from nvidia-smi
        # affinity_matrix = [
        #     range(36, 48),
        #     range(36, 48),
        #     range(12, 24),
        #     range(12, 24),
        #     range(84, 96),
        #     range(84, 96),
        #     range(60, 72),
        #     range(60, 72),
        # ]
        # os.sched_setaffinity(0, affinity_matrix[])
        NumaManager.set_affinity()
        self._gpu_store=[]
        self._offloaded = False
        self._save_event = torch.cuda.Event()
        self._resume_event = torch.cuda.Event()
        self._offload_complete_event = torch.cuda.Event()
        self._h2d_stream = h2d_stream
        self._d2h_stream = d2h_stream

        self._continuous_cpu_buffer = None
        self._continuous_gpu_buffer = None
        self._index_key_map = []
        self._index_offset = []
        self._index_cpu_buffer = []
        self._index_gpu_buffer = []

        self._recompute_tensors = {}
        self._recompute_store = []


        class SaveType(Enum):
            TENSOR=1
            PARAMETER=2
            RECOMPUTE=3
            ALIAS=4
        

        def tensor_key(tensor):
            return (tensor.shape, tensor.layout, tensor.dtype, tensor.stride())
        
        def save_tensor(tensor):
            assert not self._offloaded
            if isinstance(tensor, torch.nn.parameter.Parameter):
                return SaveType.PARAMETER, tensor

            if tensor.storage().data_ptr() in self._recompute_tensors:
                self._recompute_store.append(self._recompute_tensors[tensor.storage().data_ptr()])
                self._recompute_tensors.pop(tensor.storage().data_ptr())
                # print(f'rank {torch.distributed.get_rank()} recompute triggered for tensor {tensor.shape}')
                return SaveType.RECOMPUTE, len(self._recompute_store) - 1
            
        
            for index, stored_tensor in enumerate(self._gpu_store):
                if is_a_view(tensor, stored_tensor):
                    offset = tensor.storage_offset() - stored_tensor.storage_offset()
                    stride = tensor.stride()
                    shape = tensor.shape
                    return SaveType.ALIAS, (tensor.dtype, index, shape, stride, offset)

            self._gpu_store.append(tensor)
            if (len(self._index_key_map) < len(self._gpu_store)):
                self._index_key_map.append(tensor_key(tensor))
            else:
                assert(self._index_key_map[len(self._gpu_store) - 1] == tensor_key(tensor))
            self._save_event.record()
            # print(f"rank {torch.distributed.get_rank()} Saving tensor id {len(self._gpu_store) - 1} {id(tensor)} {tensor.shape}, dtype {tensor.dtype}, device {tensor.device} storage {tensor.storage().data_ptr()}")
            return (SaveType.TENSOR, len(self._gpu_store) - 1)
        
        def resume_tensor(packed):
            assert not self._offloaded
            if packed[0] == SaveType.PARAMETER:
                return packed[1]
            if packed[0] == SaveType.RECOMPUTE:
                p_infos, function, rng_states = self._recompute_store[packed[1]]
                parents = []
                for (dtype, index, shape, stride, offset) in p_infos:
                    if index is None:
                        parents.append(shape)
                    else:
                        self._resume_event.wait()
                        # print(f"rank {torch.distributed.get_rank()} Resuming parent tensor id {index} {shape}, offset {offset}, value {checksum(self._gpu_store[index])}")
                        parents.append(torch.as_strided(self._continuous_gpu_buffer[dtype], shape, stride, self.index_offset[index] + offset))
                with torch.no_grad():
                    if rng_states is not None:
                        current_rng_states = save_rng_states()
                        restore_rng_states(rng_states)
                    r = function(*parents)
                    if rng_states is not None:
                        restore_rng_states(current_rng_states)
                    # print(f'rank {torch.distributed.get_rank()} recomputed tensor {r.shape} {checksum(r)} inputs {[checksum(x) for x in parents]}')
                # print(f'rank {torch.distributed.get_rank()} recompute registered for tensor {r.shape}')
                return r
            if packed[0] == SaveType.ALIAS:
                dtype, index, shape, stride, offset = packed[1]
                self._resume_event.wait()
                # print(f"rank {torch.distributed.get_rank()} Resuming alias tensor id {index} {shape}, offset {offset}")
                return torch.as_strided(self._continuous_gpu_buffer[dtype], shape, stride, self.index_offset[index] + offset)

            type, id = packed
            assert type == SaveType.TENSOR
            self._resume_event.wait()
            ret = self._gpu_store[id]
            self._gpu_store[id] = None
            # print(f"rank {torch.distributed.get_rank()} Resuming tensor id {id} {ret.shape}, dtype {ret.dtype}, device {ret.device}")
            return ret
        super().__init__(save_tensor, resume_tensor)
    
    def _recompute_tensor(self, tensor, parents, function, rng_states=None):
        assert not self._offloaded
        # gpu_store_ids = [id(x.storage()) for x in self._gpu_store]
        # assert id(tensor.storage()) not in gpu_store_ids
        # print(parents[0].shape)
        
        parent_info = []       
        for parent in parents:
            found = False
            if isinstance(parent, torch.nn.parameter.Parameter):
                parent_info.append((parent.dtype, None, parent, None, None))
                found = True
                break
            for index, stored_tensor in enumerate(self._gpu_store):
                if is_a_view(parent, stored_tensor):
                    parent_info.append((parent.dtype, index, parent.shape, parent.stride(), 0))
                    found = True
                    break
            assert found, f"Parent tensor {parent.shape} not found in store"
        # print(f"rank {torch.distributed.get_rank()} Recompute registered for tensor {id(tensor)} {tensor.shape} storage {tensor.storage().data_ptr()} {checksum(tensor)} inputs {[checksum(x) for x in parents]}")
        self._recompute_tensors[tensor.storage().data_ptr()] = parent_info, function, rng_states
        
    def _allocate_buffers(self):
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
        offset=defaultdict(int)
        for (shape, layout, dtype, stride) in self._index_key_map:
            assert layout == torch.strided
            # assert dtype == torch.half, f"Only half precision supported, got {dtype} shape {shape}"
            mysize = size_of_tensor(shape, stride)
            self.index_offset.append(offset[dtype])
            offset[dtype] += mysize
        
        print(f"rank {torch.distributed.get_rank()} Allocating {offset} bytes cpu buffer")
        self._continuous_cpu_buffer = {
            dtype: torch.empty([offset], dtype=dtype, pin_memory=True, device='cpu') for dtype, offset in offset.items()
        }

        for index, (shape, layout, dtype, stride) in enumerate(self._index_key_map):
            ctensor = torch.as_strided(self._continuous_cpu_buffer[dtype], shape, stride, self.index_offset[index])
            self._index_cpu_buffer.append(ctensor)

    def _allocate_gpu_buffers(self):
        assert not self._index_gpu_buffer
        self._continuous_gpu_buffer = {
            dtype: x.to('cuda', non_blocking=True) for dtype, x in self._continuous_cpu_buffer.items()}
        for index, (shape, layout, dtype, stride) in enumerate(self._index_key_map):
            gtensor = torch.as_strided(self._continuous_gpu_buffer[dtype], shape, stride, self.index_offset[index])
            self._index_gpu_buffer.append(gtensor)

    @torch.no_grad()
    @torch.cuda.nvtx.range("Offload")
    def offload(self):
        assert not self._offloaded
        assert len(self._recompute_tensors) == 0
        # assert self._gpu_store
        size=0
        storage_size=0
        storages = set()
        # if not hasattr(self, 'placeholder'):
        #     self.placeholder = torch.ones([512, 1024, 1024], dtype=torch.half, device='cuda')
        #     self.placeholder_cpu = torch.empty_like(self.placeholder, device='cpu')
        #     self.placeholder_cpu = self.placeholder_cpu.pin_memory()
        
        # print(f"Stream: {self._stream}")
        with torch.cuda.stream(self._d2h_stream) if self._d2h_stream else contextlib.nullcontext():
            self._save_event.wait()
            self._allocate_buffers()
            for index, tensor in enumerate(self._gpu_store): # + [self.placeholder]:
                buffer = self._index_cpu_buffer[index]
                # torch.cuda.nvtx.range_push(f"D2H {buffer.shape} {buffer.layout} {buffer.dtype} {buffer.stride()}")
                buffer.copy_(tensor, non_blocking=True)
                # torch.cuda.nvtx.range_pop()
                size+=tensor.numel()
                # torch.tensor([1,2,3,4]) * torch.tensor([1,2,3,4])
                if tensor.storage().data_ptr() not in storages:
                    # print(f"rank {torch.distributed.get_rank()} Storage of tensor {tensor.shape} size {tensor.storage().size()/1000000} MB not in set")
                    storages.add(tensor.storage().data_ptr())
                    storage_size+=tensor.storage().nbytes()
                else:
                    # print(f"rank {torch.distributed.get_rank()} Storage of tensor {tensor.shape} size {tensor.storage().size()/1000000} MB already in set")
                    pass
                # print(f"Saving buffer to cpu shape {buffer.shape}, dtype {buffer.dtype}, device {buffer.device}")
            self._offload_complete_event.record()
        print(f"rank {torch.distributed.get_rank()} Offloaded {size / 1000000000} Billion elements, {len(self._gpu_store)} tensors, storage size {storage_size / 1000000000} GBytes")
        
        self._offloaded = True
        
        # torch.cuda.synchronize()
    
    def offload_release(self):
        assert self._offloaded
        if self._d2h_stream is not None:
            self._offload_complete_event.wait()
        self._recompute_tensors.clear()
        self._gpu_store.clear()


    @torch.no_grad()
    @torch.cuda.nvtx.range("Resume")
    def resume(self):
        assert self._offloaded
        original_stream = torch.cuda.current_stream()
        with torch.cuda.stream(self._h2d_stream) if self._h2d_stream else contextlib.nullcontext():
            self._allocate_gpu_buffers()
            for gtensor in self._index_gpu_buffer:
                self._gpu_store.append(gtensor)
            self._resume_event.record()
        # REMOVE:
        # self._gpu_store.clear()
        self._offloaded = False
        # torch.cuda.synchronize()
    
    def resume_release(self):
        assert all([x is None for x in self._gpu_store])
        self._resume_event.wait()
        
        # Syncback from default to side stream to allow dealloc of gpu buffers
        if self._h2d_stream is not None:
            self._h2d_stream.wait_stream(torch.cuda.current_stream())
        self._gpu_store.clear()
        self._index_gpu_buffer.clear()
        self._continuous_gpu_buffer.clear()
        self._recompute_store.clear()

        

offload_stream = None
d2h_stream = None
def get_offload_stream():
    global offload_stream
    if offload_stream is None:
        offload_stream = torch.cuda.Stream()
    return offload_stream
def get_offload_d2h_stream():
    global d2h_stream
    if d2h_stream is None:
        d2h_stream = torch.cuda.Stream()
    return d2h_stream

class ActivationStorePool:
    def __init__(self) -> None:
        self._pool = []
        self._queue = []
    
    def get_for_save(self) -> ActivationStore:
        if self._pool:
            ret = self._pool.pop(-1)
        else:
            ret = ActivationStore(get_offload_stream(), get_offload_d2h_stream())
        self._queue.append(ret)
        return ret
    
    def get_for_resume(self) -> ActivationStore:
        assert self._queue
        return self._queue.pop(0)
    
    def release(self, store):
        store.resume_release()
        self._pool.append(store)
        # print(f"Pool size {len(self._pool)}")
        # print([len(x._gpu_store) for x in self._pool])

    def is_empty(self):
        return len(self._queue) == 0
