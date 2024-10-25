import contextlib
import dataclasses
from typing import Tuple

import torch
from torch.autograd import Function
from torch.overrides import has_torch_function_unary, handle_torch_function

from megatron.core.tensor_parallel.random import get_cuda_rng_tracker, _set_cuda_rng_state


class TorchRngStates:
    def __init__(self):
        self.torch_state = torch.get_rng_state()
        self.cuda_state = None
        self.cuda_tracker_state = None
        if torch.cuda.is_available():
            self.cuda_state = torch.cuda.get_rng_state()
            self.cuda_tracker_state = get_cuda_rng_tracker().get_states()

    def restore(self):
        torch.set_rng_state(self.torch_state)
        if torch.cuda.is_available():
            # torch.cuda.set_rng_state(self.cuda_state)
            _set_cuda_rng_state(self.cuda_state)
            get_cuda_rng_tracker().set_states(self.cuda_tracker_state)

    def assert_state_equal(self, other):
        assert torch.equal(self.torch_state, other.torch_state)
        assert torch.equal(self.cuda_state, other.cuda_state)
        for k, v in self.cuda_tracker_state.items():
            assert torch.equal(v, other.cuda_tracker_state[k])

    @classmethod
    @contextlib.contextmanager
    def save_current_states(cls):
        states = cls()
        try:
            yield
        finally:
            states.restore()


def is_lazy_device(tensor):
    return tensor.device.type == "lazy"


def is_fused_kernel_acceptable(input, p):
    return (input.is_cuda or input.is_xpu or is_lazy_device(input)) \
           and 0.0 < p < 1.0 \
           and torch.ops.aten.sym_numel(input) > 0


def is_native_dropout(input, p, train: bool):
    return input.is_nested or (train and is_fused_kernel_acceptable(input, p))


def create_mask_non_native(input, p, train):
    # This is the non-fused branch implementation.
    if p == 0.0 or not train or torch.ops.aten.sym_numel(input) == 0:
        return torch.ones_like(input)
    if p == 1.0:
        return torch.zeros_like(input)
    # The memory format in the C++ codes is LEGACY_CONTIGUOUS_MEMORY_FORMAT:
    #       define LEGACY_CONTIGUOUS_MEMORY_FORMAT c10::get_contiguous_memory_format()
    noise = torch.empty_like(input, memory_format=torch.contiguous_format)
    noise.bernoulli_(1 - p)
    # noise.div_(1 - p)  # divide later
    return noise


def dropout_with_mask(input, p, train: bool):
    # This is a python re-implementation of dropout in Dropout.cpp
    if is_native_dropout(input, p, train):
        # Need to call native_dropout to make results match torch.nn.functional.dropout
        output, mask = torch.ops.aten.native_dropout(input, p, train)
    else:
        # The non-native implementation of torch.nn.functional.dropout is defined in _dropout_impl in Dropout.cpp
        mask = create_mask_non_native(input, p, train)
        output = input * mask / (1 - p)
    return output, mask


class RecomputedDropout(Function):
    @staticmethod
    def forward(ctx, input, p=0.5, train=True):
        ctx.p = p
        ctx.train = train
        ctx.rng_states = TorchRngStates()
        # ctx.rng_states.assert_state_equal(TorchRngStates())
        ctx.input_attrs = get_tensor_attributes(input)

        # Not to call ctx.save_for_backward for mask here,
        # then mask won't be saved for activation
        # and will not trigger the pack_hook in saved_tensors_hooks
        output, mask = dropout_with_mask(input, p, train)

        # For debug
        # ctx.mask = mask
        return output

    @staticmethod
    def backward(ctx, grad_output):
        # Restore random states to generate the same mask
        with TorchRngStates.save_current_states():
            ctx.rng_states.restore()
            # ctx.rng_states.assert_state_equal(TorchRngStates())

            # Not to access ctx.saved_tensors for mask here,
            # then the mask will not trigger the unpack_hook in saved_tensors_hooks
            assert ctx.train
            # Just need to fill something as input.
            # The data of dummy_input does matter but the memory attributes have to be the same.
            # Cannot use grad_output as input or the mask will be wrong because grad_output is not contiguous.
            assert ctx.input_attrs.memory_format is not None, \
                "memory_format of input must be contiguous or we can't reproduce the same mask value in backward"
            dummy_input = torch.empty_like(grad_output)
            dummy_input_attrs = get_tensor_attributes(dummy_input)
            assert dummy_input_attrs == ctx.input_attrs
            _, mask = dropout_with_mask(dummy_input, ctx.p, True)

        # assert torch.equal(mask, ctx.mask)
        # Reuse the memory of dummy_input
        grad_input = torch.multiply(grad_output, mask, out=dummy_input)
        grad_input = grad_input.div_(1 - ctx.p)
        # Return gradient for input and None for dropout probability
        return grad_input, None, None


def dropout(input, p=0.5, training=True):
    if has_torch_function_unary(input):
        return handle_torch_function(dropout, (input,), input, p=p, training=training, inplace=False)
    if p < 0.0 or p > 1.0:
        raise ValueError(f"dropout probability has to be between 0 and 1, but got {p}")
    return RecomputedDropout.apply(input, p, training)


@dataclasses.dataclass(eq=True, frozen=True)
class TensorAttrs:
    shape: torch.Size
    dtype: torch.dtype
    device: torch.device
    layout: torch.layout
    stride: Tuple[int, ...]
    memory_format: torch.memory_format


def get_tensor_attributes(tensor) -> TensorAttrs:
    return TensorAttrs(
        shape=tensor.shape,
        dtype=tensor.dtype,
        device=tensor.device,
        layout=tensor.layout,
        stride=tensor.stride(),
        memory_format=get_memory_format(tensor),
    )


def get_memory_format(tensor):
    if tensor.is_contiguous(memory_format=torch.channels_last):
        return torch.channels_last
    elif tensor.is_contiguous(memory_format=torch.channels_last_3d):
        return torch.channels_last_3d
    elif tensor.is_contiguous():
        return torch.contiguous_format
    else:
        # non contiguous
        return None
