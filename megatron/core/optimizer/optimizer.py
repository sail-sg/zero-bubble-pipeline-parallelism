# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

"""Megatron optimizer."""

import functools
import copy
import math
from abc import ABC, abstractmethod
from itertools import chain
from logging import getLogger
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from megatron.core import mpu
from megatron.training import get_args
from megatron.training.utils import nvtx_profile
from megatron.core.optimizer.zbpp_optimizer_helper import rollback_optimizer_step

import torch

try:
    from transformer_engine.pytorch.optimizers import multi_tensor_applier, multi_tensor_scale
except ImportError:
    try:
        from apex.multi_tensor_apply import multi_tensor_applier
    except ImportError:
        from megatron.core.utils import local_multi_tensor_applier

        multi_tensor_applier = local_multi_tensor_applier
    try:
        import amp_C

        l2_norm_impl = amp_C.multi_tensor_l2norm
        multi_tensor_scale_impl = amp_C.multi_tensor_scale
    except ImportError:
        from megatron.core.utils import local_multi_tensor_l2_norm, local_multi_tensor_scale

        l2_norm_impl = local_multi_tensor_l2_norm
        multi_tensor_scale_impl = local_multi_tensor_scale

from .. import parallel_state, tensor_parallel
from ..dist_checkpointing.mapping import ShardedStateDict
from ..dist_checkpointing.optimizer import (
    get_param_id_to_sharded_param_map,
    make_sharded_optimizer_tensor,
    optim_state_to_sharding_state,
)
from ..dist_checkpointing.utils import add_prefix_for_sharding
from ..transformer.module import param_is_not_shared
from .clip_grads import clip_grad_by_total_norm_fp32, count_zeros_fp32, get_grad_norm_fp32
from .grad_scaler import MegatronGradScaler
from .optimizer_config import OptimizerConfig

logger = getLogger(__name__)


def _zero_grad_group_helper(group: List[torch.nn.Parameter], set_to_none: bool):
    """
    Zero out the gradient for a group of parameters.
    Note: copied from torch.optim.optimizer.
    """
    for param in group:
        if param.grad is not None:
            if set_to_none:
                param.grad = None
            else:
                if param.grad.grad_fn is not None:
                    param.grad.detach_()
                else:
                    param.grad.requires_grad_(False)
                param.grad.zero_()


def _multi_tensor_copy_this_to_that(
    this: List[torch.Tensor], that: List[torch.Tensor], overflow_buf: Optional[torch.Tensor] = None
):
    """
    Use multi-tensor-applier to copy values from one list to another.
    We don't have a bfloat16 implementation so for now if the overflow_buf
    is not provided, we default back to simple loop copy to be compatible
    with bfloat16.
    """
    if overflow_buf:
        overflow_buf.fill_(0)
        # Scaling with factor `1.0` is equivalent to copy.
        multi_tensor_applier(multi_tensor_scale_impl, overflow_buf, [this, that], 1.0)
    else:
        for this_, that_ in zip(this, that):
            that_.copy_(this_)


class MegatronOptimizer(ABC):
    """
    Base class for all Megatron optimizers.

    Args:
        optimizer (torch.optim.Optimizer): base optimizer such as Adam or SGD.
        config (OptimizerConfig): configuration object for optimizer.
        init_state_fn (Callable, optional): function to initialize state in the optimizer.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        config: OptimizerConfig,
        init_state_fn: Callable = lambda x: None,
    ):
        """Input optimizer is the base optimizer (e.g., Adam)."""
        self.optimizer = optimizer
        assert self.optimizer, 'no optimizer is provided.'
        self.config = config
        self.init_state_fn = init_state_fn

        self.partial_reduced_total_norm = torch.FloatTensor([0])
        self.local_total_norm = None
        self.dummy_overflow_buf = torch.cuda.IntTensor([0])
        self.zero_float_tensor = torch.cuda.FloatTensor([0])
        self.parameters_backup = None
        self.do_prev_step = False
        self.do_this_step = False
        self.send_next_reqs = []
        self.send_prev_reqs = []
        self.grad_norm_no_clip_recorder = 0
        self.post_validation_enabled = False

    def record_grad_norm(self, grad_norm):
        if self.post_validation_enabled:
            return
        if self.config.clip_grad > 0.0:
            if grad_norm is None or grad_norm > self.config.clip_grad:
                self.grad_norm_no_clip_recorder = 0
            else:
                self.grad_norm_no_clip_recorder += 1
            if self.grad_norm_no_clip_recorder >= 10:
                rank = parallel_state.get_pipeline_model_parallel_rank()
                print(f"{rank}: enable optimizer post validation")
                self.post_validation_enabled = True
        else:
            if grad_norm is not None:
                # optimizer state update successfully
                rank = parallel_state.get_pipeline_model_parallel_rank()
                print(f"{rank}: enable optimizer post validation")
                self.post_validation_enabled = True

    @torch.no_grad()
    def save_parameters_backup(self):
        parameters = self.get_parameters()
        backups = []
        for param in parameters:
            p = param.detach().clone()
            s1 = self.optimizer.state[param]["exp_avg"].detach().clone() if "exp_avg" in self.optimizer.state[param] else torch.zeros_like(param.data).float()
            s2 = self.optimizer.state[param]["exp_avg_sq"].detach().clone() if "exp_avg_sq" in self.optimizer.state[param] else torch.zeros_like(param.data).float()
            backups.append((p, s1, s2))
        self.parameters_backup = backups

    @torch.no_grad()
    def rollback_parameters(self):
        parameters = self.get_parameters()
        for param, (backup, s1, s2) in zip(parameters, self.parameters_backup):
            param.copy_(backup)
            self.optimizer.state[param]["exp_avg"] = s1
            self.optimizer.state[param]["exp_avg_sq"] = s2
        self.parameters_backup = None

    def get_mp_group_except_pp_for_bypassing_sync(self):
        """Default returned here, but the distributed optimizer overrides this."""
        # Note: expert parallel are not supported yet
        return mpu.get_tensor_model_parallel_group()

    def calc_local_grad_norm(self):
        grads_for_norm = self.get_main_grads_for_grad_norm()
        return self.do_clac_local_grad_norm(
            grads_for_norm,
            tensor_parallel_group=self.get_mp_group_except_pp_for_bypassing_sync())

    def get_clip_coeff_and_grad_norm(self, max_norm, norm_type=2):
        _total_norm = self.partial_reduced_total_norm
        if norm_type == torch.inf:
            _total_norm = _total_norm[0].item()
        else:
            _total_norm = _total_norm.item() ** (1.0 / norm_type)
        _clip_coeff = max_norm / (_total_norm + 1.0e-6)
        return _clip_coeff, _total_norm

    def do_clac_local_grad_norm(
        self, grads_for_norm, norm_type=2,
        tensor_parallel_group=None
    ):
        if isinstance(grads_for_norm, torch.Tensor):
            grads_for_norm = [grads_for_norm]

        # Norm parameters.
        norm_type = float(norm_type)
        total_norm = 0.0

        # Calculate norm.
        if norm_type == torch.inf:
            total_norm = max(grad.abs().max() for grad in grads_for_norm)
            total_norm_cuda = torch.cuda.FloatTensor([float(total_norm)])
            # Take max across all model-parallel GPUs.
            torch.distributed.all_reduce(total_norm_cuda,
                                         op=torch.distributed.ReduceOp.MAX,
                                         group=tensor_parallel_group)
            total_norm = total_norm_cuda
            # total_norm = total_norm_cuda[0].item()

        else:
            if norm_type == 2.0:
                self.dummy_overflow_buf.fill_(0)
                # Use apex's multi-tensor applier for efficiency reasons.
                # Multi-tensor applier takes a function and a list of list
                # and performs the operation on that list all in one kernel.
                if grads_for_norm:
                    import amp_C
                    grad_norm, _ = multi_tensor_applier(
                        amp_C.multi_tensor_l2norm,
                        self.dummy_overflow_buf,
                        [grads_for_norm],
                        False  # no per-parameter norm
                    )
                else:
                    self.zero_float_tensor.fill_(0)
                    grad_norm = self.zero_float_tensor
                # Since we will be summing across data parallel groups,
                # we need the pow(norm-type).
                total_norm = grad_norm ** norm_type

            else:
                for grad in grads_for_norm:
                    grad_norm = torch.norm(grad, norm_type)
                    total_norm += grad_norm ** norm_type

            # Sum across all model-parallel GPUs.
            torch.distributed.all_reduce(total_norm,
                                         op=torch.distributed.ReduceOp.SUM,
                                         group=tensor_parallel_group)
            # total_norm = total_norm.item() ** (1.0 / norm_type)

        self.local_total_norm = total_norm.cpu()
        return total_norm

    def partially_reduce_local_total_norm(self, clip_grad):
        return self.do_partially_reduce_local_total_norm(clip_grad)

    def do_partially_reduce_local_total_norm(self, max_norm, norm_type=2):
        # recv value from prev pipeline stage
        # self.partial_reduced_total_norm = self.recv_one(self.partial_reduced_total_norm)
        prev_clip_coeff, prev_grad_norm = self.get_clip_coeff_and_grad_norm(max_norm, norm_type)

        # reduce
        if norm_type == torch.inf:
            self.partial_reduced_total_norm = torch.maximum(self.partial_reduced_total_norm, self.local_total_norm)
        else:
            self.partial_reduced_total_norm = self.partial_reduced_total_norm + self.local_total_norm

        this_clip_coeff, grad_norm = self.get_clip_coeff_and_grad_norm(max_norm, norm_type)
        # rank = parallel_state.get_pipeline_model_parallel_rank()
        return prev_clip_coeff, this_clip_coeff, grad_norm

    def downscale_gradient(self, clip_coeff):
        assert clip_coeff < 1.0
        parameters = self.get_parameters()
        if isinstance(parameters, torch.Tensor):
            parameters = [parameters]
        # Grads.
        grads = []
        for param in parameters:
            if param.grad is not None:
                assert param.grad.type() == 'torch.cuda.FloatTensor'
                grads.append(param.grad.detach())
        self.dummy_overflow_buf.fill_(0)
        multi_tensor_applier(amp_C.multi_tensor_scale,
                             self.dummy_overflow_buf,
                             [grads, grads],
                             clip_coeff)

    def get_reduced_global_states(self):
        return [self.partial_reduced_total_norm]

    def send_all(self, to_next=True):
        need_send = False
        dst = None
        if to_next and not parallel_state.is_pipeline_last_stage(ignore_virtual=True):
            need_send = True
            dst = parallel_state.get_pipeline_model_parallel_next_rank()
        if not to_next and not parallel_state.is_pipeline_first_stage(ignore_virtual=True):
            need_send = True
            dst = parallel_state.get_pipeline_model_parallel_prev_rank()
        if need_send:
            for global_state in self.get_reduced_global_states():
                send_req = torch.distributed.isend(
                    tensor=global_state,
                    dst=dst,
                    group=parallel_state.get_pipeline_model_parallel_group(),
                )
                if to_next:
                    self.send_next_reqs.append(send_req)
                else:
                    self.send_prev_reqs.append(send_req)

    def recv_all(self, from_prev=True, init_values=None):
        if from_prev:
            for req in self.send_prev_reqs:
                req.wait()
            self.send_prev_reqs = []
        else:
            for req in self.send_next_reqs:
                req.wait()
            self.send_next_reqs = []
        all_global_states = self.get_reduced_global_states()
        if init_values is None:
            init_values = [0.0] * len(all_global_states)
        for global_state, init_value in zip(all_global_states, init_values):
            self.recv_one(global_state, from_prev=from_prev, init_value=init_value)

    def recv_one(self, global_state, from_prev=True, init_value=0.0):
        if from_prev:
            if parallel_state.is_pipeline_first_stage(ignore_virtual=True):
                global_state.fill_(init_value)
            else:
                req = torch.distributed.irecv(
                    tensor=global_state,
                    src=parallel_state.get_pipeline_model_parallel_prev_rank(),
                    group=parallel_state.get_pipeline_model_parallel_group(),
                )
                req.wait()
        else:
            if not parallel_state.is_pipeline_last_stage(ignore_virtual=True):
                req = torch.distributed.irecv(
                    tensor=global_state,
                    src=parallel_state.get_pipeline_model_parallel_next_rank(),
                    group=parallel_state.get_pipeline_model_parallel_group(),
                )
                req.wait()
        return global_state

    def get_parameters(self) -> List[torch.nn.Parameter]:
        """
        Get list of parameters wrapped in optimizer.
        """
        params = []
        for param_group in self.optimizer.param_groups:
            for param in param_group['params']:
                params.append(param)
        return params

    def get_main_grads_for_grad_norm(self) -> List[torch.Tensor]:
        """
        Get main_grads that should be taken into account to compute the grad norm.
        Filter parameters based on:
          - grad should not be None.
          - parameter should not be shared (i.e., grads shouldn't be double counted while
            computing norms).
          - should not be a replica due to tensor model parallelism.
        """
        params = self.get_parameters()
        grads_for_norm = []
        for param in params:
            grad = param.grad
            grad_not_none = grad is not None
            is_not_shared = param_is_not_shared(param)
            is_not_tp_duplicate = tensor_parallel.param_is_not_tensor_parallel_duplicate(param)
            if grad_not_none and is_not_shared and is_not_tp_duplicate:
                grads_for_norm.append(grad)

        return grads_for_norm

    def get_model_parallel_group(self) -> torch.distributed.ProcessGroup:
        """Default returned here, but the distributed optimizer overrides this."""
        if hasattr(self, 'model_parallel_group'):
            return self.model_parallel_group
        return parallel_state.get_model_parallel_group()

    @abstractmethod
    def prepare_grads(self) -> bool:
        """Pre-processing gradients before the optimizer step, returns whether inf/nan is found."""
        return False

    @abstractmethod
    def step_with_ready_grads(self) -> bool:
        """Step the optimizer with ready gradients, return successful."""
        return True

    @torch.no_grad()
    def get_grad_norm(self):
        grads_for_norm = self.get_main_grads_for_grad_norm()
        total_norm = get_grad_norm_fp32(
            grads_for_norm,
            model_parallel_group=self.get_model_parallel_group(),
        )
        return total_norm

    def clip_grad_norm(self, clip_grad: float) -> float:
        """Compute grad norm."""
        params = self.get_parameters()
        grads_for_norm = self.get_main_grads_for_grad_norm()
        grad_norm = get_grad_norm_fp32(
            grads_for_norm, model_parallel_group=self.get_model_parallel_group()
        )
        clip_grad_by_total_norm_fp32(params, clip_grad, grad_norm)
        return grad_norm

    def count_zeros(self) -> float:
        """Count number of zeros in model's gradients."""
        params = self.get_parameters()
        return count_zeros_fp32(params, model_parallel_group=self.get_model_parallel_group())

    @abstractmethod
    def zero_grad(self, set_to_none: bool = True):
        pass

    @abstractmethod
    def get_loss_scale(self) -> torch.Tensor:
        """
        Get current loss scale factor.
        NOTE: The output should be a CUDA tensor of size 1.
        """
        pass

    def scale_loss(self, loss: torch.Tensor) -> torch.Tensor:
        """Simple scaling."""
        return self.get_loss_scale() * loss

    def finish_param_sync(self, model_index: int):
        """
        Finish parameter synchronization for all optimizers.
        This is a no-op for all non-distributed optimizers.
        """
        pass

    @abstractmethod
    def reload_model_params(self):
        """Refreshes any internal state from the current model parameters.
        Call whenever the parameters are changed outside of the optimizer.
        For example, when we load a model from a checkpoint  without loading
        the optimizer, the model parameters are updated but for fp16 optimizer
        with main parameters, the main parameters need to also be updated."""
        pass

    @abstractmethod
    def state_dict(self):
        pass

    @abstractmethod
    def load_state_dict(self, state_dict):
        pass

    # Promote state so it can be retrieved or set via
    # "optimizer_instance.state"
    def _get_state(self):
        return self.optimizer.state

    def _set_state(self, value):
        self.optimizer.state = value

    state = property(_get_state, _set_state)

    # Promote param_groups so it can be retrieved or set via
    # "optimizer_instance.param_groups"
    # (for example, to adjust the learning rate)
    def _get_param_groups(self):
        return self.optimizer.param_groups

    def _set_param_groups(self, value):
        self.optimizer.param_groups = value

    param_groups = property(_get_param_groups, _set_param_groups)

    @abstractmethod
    def step(self):
        """Step the optimizer."""
        pass

    @abstractmethod
    def sharded_state_dict(
        self, model_sharded_state_dict: ShardedStateDict, is_loading: bool = False
    ) -> ShardedStateDict:
        """Builds sharded state dict for the optimizer, based on model's sharded state dict.

        Args:
            model_sharded_state_dict (ShardedStateDict): sharded state dict of the model
            is_loading (bool, optional): flag indicating whether the state dict will be used to save or load the optimizer state.
                Defaults to False.

        Returns: optimizer sharded state dict
        """

    @staticmethod
    def _extract_common_per_param_step(state_dict) -> Union[int, torch.Tensor]:
        common_step = None
        for param_idx, param_state in state_dict['state'].items():
            param_step = param_state.get('step', None)
            if param_step is not None:
                if common_step is None:
                    common_step = param_step
                elif common_step != param_step:
                    raise ValueError(
                        "The optimizer step differs per parameter. Mcore only supports "
                        "optimizers whose step is shared across all parameters."
                    )
        return common_step

    @staticmethod
    def _restore_common_per_param_step(state_dict: Dict, step: Union[int, torch.Tensor]):
        for param_idx, param_state in state_dict['state'].items():
            param_state['step'] = copy.deepcopy(step)


class MixedPrecisionOptimizer(MegatronOptimizer):
    """Base class for both the float-16 and the distributed optimizer.

    Args:
        optimizer (torch.optim.Optimizer): base optimizer such as Adam or SGD.
        config (OptimizerConfig): configuration object for optimizer.
        grad_scaler (MegatronGradScaler): used for scaling gradients. Note that
            this can be None. This case happens when `bf16 = True` and we don't
            use any loss scale. Note that for `bf16 = True`, we can have
            a constant gradient scaler. Also for `bf16 = False`, we
            always require a grad scaler.
        init_state_fn (Callable, optional): function to initialize state in the optimizer.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        config: OptimizerConfig,
        grad_scaler: Optional[MegatronGradScaler],
        init_state_fn: Callable,
    ):

        super().__init__(
            optimizer,
            config,
            init_state_fn,
        )
        self.grad_scaler = grad_scaler

        # None grad scaler is only supported for bf16.
        if self.grad_scaler is None:
            assert not self.config.fp16, 'fp16 expects a grad scaler.'

        # Tensor used to determine if a nan/if has happend.
        # Any non-zero value indicates inf/nan.
        # Note that we keep this for the cases that grad scaler is none.
        # We still record nan/inf if we have a bfloat16 with a grad scaler.
        if self.grad_scaler:
            self.found_inf = torch.tensor([0.0], dtype=torch.float, device='cuda')
            self.partial_reduced_found_inf = torch.FloatTensor([0.0])
        self.fully_reduced_global_states = None

        # Dummy tensor needed for apex multi-apply tensor.
        # For bfloat, we don't have multi-tensor apply and for now
        # we set it to none so the multi-tensor apply gets ignored.
        if self.config.bf16:
            self._dummy_overflow_buf = None
        else:
            self._dummy_overflow_buf = torch.tensor([0], dtype=torch.int, device='cuda')

        # In case grad scaler is not passed, define the unity scale.
        if self.grad_scaler is None:
            self._scale_one = torch.tensor([1.0], dtype=torch.float, device='cuda')

    def get_reduced_global_states(self):
        reduced_global_states = []
        if self.grad_scaler:
            reduced_global_states.append(self.partial_reduced_found_inf)
        reduced_global_states.extend(super().get_reduced_global_states())
        return reduced_global_states

    def get_found_inf_flag(self):
        return self.partial_reduced_found_inf.item() > 0

    def _local_unscale_main_grads_and_check_for_nan(self):
        # Collect main grads.
        main_grads = self._collect_main_grad_data_for_unscaling()
        # Reset found inf.
        self.found_inf.fill_(0.0)
        # Unscale and set found inf/nan
        torch._amp_foreach_non_finite_check_and_unscale_(
            main_grads, self.found_inf, self.grad_scaler.inv_scale)
        # Update across all model parallel instances, except pipeline parallel
        torch.distributed.all_reduce(self.found_inf,
                                     op=torch.distributed.ReduceOp.MAX,
                                     group=self.get_mp_group_except_pp_for_bypassing_sync())

    def partially_reduce_local_found_inf(self):
        # self.partial_reduced_found_inf = self.recv_one(self.partial_reduced_found_inf)
        # check for nan in previous rank
        prev_found_inf_flag = self.get_found_inf_flag()
        self.partial_reduced_found_inf = torch.maximum(self.partial_reduced_found_inf, self.found_inf.cpu())
        # Check for nan.
        this_found_inf_flag = self.get_found_inf_flag()
        return prev_found_inf_flag, this_found_inf_flag

    @functools.partial(nvtx_profile, name="recv_pre_step")
    @torch.no_grad()
    def recv_pre_step(self):
        # recv global states to prev rank
        self.recv_all()

    @functools.partial(nvtx_profile, name="send_pre_step")
    @torch.no_grad()
    def send_pre_step(self):
        # send global states to next rank
        self.send_all()

    @functools.partial(nvtx_profile, name="pre_step")
    @torch.no_grad()
    def pre_step(self, args, timers):
        # Copy gradients from model params to main params.
        timers('optimizer-copy-to-main-grad', log_level=1).start(
            barrier=args.barrier_with_L1_time)
        self._copy_model_grads_to_main_grads()
        timers('optimizer-copy-to-main-grad').stop()

        rank = parallel_state.get_pipeline_model_parallel_rank()

        if self.grad_scaler:
            self._local_unscale_main_grads_and_check_for_nan()
        if self.config.clip_grad > 0.0:
            local_norm = self.calc_local_grad_norm()

        # recv global states to prev rank
        # self.recv_all()
        self.recv_pre_step()
        prev_found_inf_flag, this_found_inf_flag = False, False
        if self.grad_scaler:
            # Unscale and check for inf/nan.
            timers('optimizer-unscale-and-check-inf', log_level=1).start(
                barrier=args.barrier_with_L1_time)
            prev_found_inf_flag, this_found_inf_flag = self.partially_reduce_local_found_inf()
            timers('optimizer-unscale-and-check-inf').stop()

        # Clip the main gradients.
        timers('optimizer-reduce-grad-norm', log_level=1).start(
            barrier=args.barrier_with_L1_time)
        grad_norm = None
        prev_clip_coeff, this_clip_coeff = 2.0, 2.0
        if self.config.clip_grad > 0.0:
            prev_clip_coeff, this_clip_coeff, grad_norm = self.partially_reduce_local_total_norm(self.config.clip_grad)
        timers('optimizer-reduce-grad-norm').stop()

        # send global states to next rank
        # self.send_all()
        self.send_pre_step()

        def can_local_step(found_inf_flag, clip_coeff):
            if self.grad_scaler:
                if found_inf_flag:
                    return False
            if self.config.clip_grad > 0.0:
                is_nan = clip_coeff == float('inf') or \
                         clip_coeff == -float('inf') or \
                         clip_coeff != clip_coeff
                assert not is_nan
                if is_nan or clip_coeff < 1.0:
                    return False
            return True
        self.do_prev_step = can_local_step(prev_found_inf_flag, prev_clip_coeff)
        self.do_this_step = can_local_step(this_found_inf_flag, this_clip_coeff)
        # print(f"{rank} pre_step: {prev_found_inf_flag}, {prev_clip_coeff} -> {self.do_prev_step} | {this_found_inf_flag}, {this_clip_coeff} -> {self.do_this_step}")
        timers('optimizer-local-step', log_level=1).start(
            barrier=args.barrier_with_L1_time)
        if self.do_this_step:
            # Step the optimizer.
            if args.enable_exactly_numeric_match:
                self.save_parameters_backup()  # for exactly match
            self.optimizer.step()
        timers('optimizer-local-step').stop()

        # Update params from main params.
        timers('optimizer-copy-main-to-model-params', log_level=1).start(
            barrier=args.barrier_with_L1_time)
        if self.do_this_step:
            self._copy_main_params_to_model_params()
        timers('optimizer-copy-main-to-model-params').stop()
        if self.do_this_step:
            self._release_grad_fp32_from_fp16()

    def prepare_fully_reduced_global_states(self):
        self.fully_reduced_global_states = {}
        if self.grad_scaler:
            found_inf_flag = self.get_found_inf_flag()
            self.fully_reduced_global_states["found_inf_flag"] = found_inf_flag
        if self.config.clip_grad > 0.0:
            clip_coeff, grad_norm = self.get_clip_coeff_and_grad_norm(self.config.clip_grad)
            self.fully_reduced_global_states["clip_coeff"] = clip_coeff
            self.fully_reduced_global_states["grad_norm"] = grad_norm

    @functools.partial(nvtx_profile, name="RECV_POST_VALIDATION")
    @torch.no_grad()
    def recv_post_validation(self):
        self.recv_all(from_prev=False)
        if parallel_state.is_pipeline_first_stage(ignore_virtual=True):
            self.prepare_fully_reduced_global_states()

    @functools.partial(nvtx_profile, name="SEND_POST_VALIDATION")
    @torch.no_grad()
    def send_post_validation(self):
        if not parallel_state.is_pipeline_first_stage(ignore_virtual=True):
            self.prepare_fully_reduced_global_states()
        self.send_all(to_next=False)

    @torch.no_grad()
    def recompute_fp32_grad(self):
        self._copy_model_grads_to_main_grads()
        if self.grad_scaler:
            # Collect fp32 main grads from fp16.
            main_grads = self._collect_main_grad_data_for_unscaling()
            # Reset found inf.
            self.found_inf.fill_(0.0)
            # Unscale and set found inf/nan
            torch._amp_foreach_non_finite_check_and_unscale_(
                main_grads, self.found_inf, self.grad_scaler.inv_scale)

    @functools.partial(nvtx_profile, name="POST_VALIDATION")
    @torch.no_grad()
    def post_validation(self, free_buffers_callback):
        rank = parallel_state.get_pipeline_model_parallel_rank()
        global_rank = torch.distributed.get_rank()
        if self.grad_scaler:
            # found_inf_flag = self.get_found_inf_flag()
            found_inf_flag = self.fully_reduced_global_states["found_inf_flag"]
            if found_inf_flag:
                if self.do_this_step:
                    print(f"{rank}-{global_rank} found inf rollback")
                    free_buffers_callback()
                    self.recompute_fp32_grad()
                    rollback_optimizer_step(self.optimizer)
                    if get_args().enable_exactly_numeric_match:
                        self.rollback_parameters()  # for exactly match
                    self._copy_main_params_to_model_params()
                self.grad_scaler.update(found_inf_flag)
                return False, None, self.do_this_step, False
            self.grad_scaler.update(found_inf_flag)
        succeed = True
        grad_norm = None
        if self.config.clip_grad > 0.0:
            # clip_coeff, grad_norm = self.get_clip_coeff_and_grad_norm(self.clip_grad)
            clip_coeff, grad_norm = self.fully_reduced_global_states["clip_coeff"], self.fully_reduced_global_states["grad_norm"]
            is_nan = clip_coeff == float('inf') or \
                     clip_coeff == -float('inf') or \
                     clip_coeff != clip_coeff
            assert not is_nan
            if clip_coeff < 1.0:
                if self.do_this_step:
                    print(f"{rank}-{global_rank} grad rollback {clip_coeff}")
                    free_buffers_callback()
                    self.recompute_fp32_grad()
                    rollback_optimizer_step(self.optimizer)
                    if get_args().enable_exactly_numeric_match:
                        self.rollback_parameters()  # for exactly match
                if get_args().enable_exactly_numeric_match:
                    clip_coeff = round(clip_coeff, 4)  # for exactly match
                self.downscale_gradient(clip_coeff)
                self.optimizer.step()
                self._copy_main_params_to_model_params()
                succeed = False
            else:
                assert self.do_this_step
        else:
            assert self.do_this_step

        # updated, grad_norm, rollback, succeed
        return True, grad_norm, not succeed and self.do_this_step, succeed

    def get_loss_scale(self):
        if self.grad_scaler is None:
            return self._scale_one
        return self.grad_scaler.scale

    def reload_model_params(self):
        self._copy_model_params_to_main_params()

    def _unscale_main_grads_and_check_for_nan(self):

        # Collect main grads.
        main_grads = self._collect_main_grad_data_for_unscaling()

        # Reset found inf.
        self.found_inf.fill_(0.0)

        # Unscale and set found inf/nan
        torch._amp_foreach_non_finite_check_and_unscale_(
            main_grads, self.found_inf, self.grad_scaler.inv_scale
        )

        # Update across all model parallel instances.
        torch.distributed.all_reduce(
            self.found_inf, op=torch.distributed.ReduceOp.MAX, group=self.get_model_parallel_group()
        )

        # Check for nan.
        found_inf_flag = self.found_inf.item() > 0

        return found_inf_flag

    @torch.no_grad()
    def prepare_grads(self) -> bool:
        """Pre-processing gradients before the optimizer step, returns whether inf/nan is found."""
        timers = self.config.timers

        # Copy gradients from model params to main params.
        if timers is not None:
            timers('optimizer-copy-to-main-grad', log_level=1).start(
                barrier=self.config.barrier_with_L1_time
            )
        self._copy_model_grads_to_main_grads()
        if timers is not None:
            timers('optimizer-copy-to-main-grad').stop()

        # Do unscale, check for inf, and update grad scaler only for
        # the case that grad scaler is provided.
        if self.grad_scaler:

            # Unscale and check for inf/nan.
            if timers is not None:
                timers('optimizer-unscale-and-check-inf', log_level=1).start(
                    barrier=self.config.barrier_with_L1_time
                )
            found_inf_flag = self._unscale_main_grads_and_check_for_nan()
            if timers is not None:
                timers('optimizer-unscale-and-check-inf').stop()

            # We are done with scaling gradients
            # so we can update the loss scale.
            self.grad_scaler.update(found_inf_flag)

            return found_inf_flag

        return False

    @torch.no_grad()
    def step_with_ready_grads(self) -> bool:
        """Step the optimizer with ready gradients, return successful."""
        timers = self.config.timers
        # Step the optimizer.
        if timers is not None:
            timers('optimizer-inner-step', log_level=1).start(
                barrier=self.config.barrier_with_L1_time
            )
        self.optimizer.step()
        if timers is not None:
            timers('optimizer-inner-step').stop()

        # Update params from main params.
        if timers is not None:
            timers('optimizer-copy-main-to-model-params', log_level=1).start(
                barrier=self.config.barrier_with_L1_time
            )
        self._copy_main_params_to_model_params()
        if timers is not None:
            timers('optimizer-copy-main-to-model-params').stop()

        return True

    @torch.no_grad()
    def step(self):
        timers = self.config.timers

        found_inf_flag = self.prepare_grads()
        if found_inf_flag:
            return False, None, None

        # Clip the main gradients.
        if timers is not None:
            timers('optimizer-clip-main-grad', log_level=1).start(
                barrier=self.config.barrier_with_L1_time
            )
        grad_norm = None
        if self.config.clip_grad > 0.0:
            grad_norm = self.clip_grad_norm(self.config.clip_grad)
        if timers is not None:
            timers('optimizer-clip-main-grad').stop()

        # Count the zeros in the grads.
        if timers is not None:
            timers('optimizer-count-zeros', log_level=1).start(
                barrier=self.config.barrier_with_L1_time
            )
        num_zeros_in_grad = self.count_zeros() if self.config.log_num_zeros_in_grad else None
        if timers is not None:
            timers('optimizer-count-zeros').stop()

        success = self.step_with_ready_grads()

        # Successful update.
        return success, grad_norm, num_zeros_in_grad


class Float16OptimizerWithFloat16Params(MixedPrecisionOptimizer):
    """Float16 optimizer for fp16 and bf16 data types.

    Args:
        optimizer (torch.optim.Optimizer): base optimizer such as Adam or SGD.
        config (OptimizerConfig): configuration object for optimizer.
        grad_scaler (MegatronGradScaler): used for scaling gradients. Note that
            this can be None. This case happens when `bf16 = True` and we don't
            use any loss scale. Note that for `bf16 = True`, we can have
            a constant gradient scaler. Also for `bf16 = False`, we
            always require a grad scaler.
        init_state_fn (Callable, optional): function to initialize state in the optimizer.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        config: OptimizerConfig,
        grad_scaler: MegatronGradScaler,
        init_state_fn: Callable,
    ):

        super().__init__(
            optimizer,
            config,
            grad_scaler,
            init_state_fn,
        )

        # Handle main parameters.

        # Three groups of parameters:
        #   float16_groups: original float16 parameters
        #   fp32_from_float16_groups: fp32 copy of float16 parameters
        #   fp32_from_fp32_groups: original fp32 parameters
        self.float16_groups = []
        self.fp32_from_float16_groups = []
        self.fp32_from_fp32_groups = []

        # For all the groups in the original optimizer:
        for param_group in self.optimizer.param_groups:
            float16_params_this_group = []
            fp32_params_this_group = []
            fp32_from_float16_params_this_group = []
            # For all the parameters in this group:
            for i, param in enumerate(param_group['params']):
                if param.requires_grad:

                    # float16 params:
                    if param.type() in ['torch.cuda.HalfTensor', 'torch.cuda.BFloat16Tensor']:
                        float16_params_this_group.append(param)
                        # Create a copy
                        main_param = param.detach().clone().float()
                        # Copy tensor model parallel attributes.
                        tensor_parallel.copy_tensor_model_parallel_attributes(main_param, param)
                        if hasattr(param, 'shared'):
                            main_param.shared = param.shared
                        # Replace the optimizer params with the new fp32 copy.
                        param_group['params'][i] = main_param

                        fp32_from_float16_params_this_group.append(main_param)
                        # Reset existing state dict key to the new main param.
                        if param in self.optimizer.state:
                            self.optimizer.state[main_param] = self.optimizer.state.pop(param)
                    # fp32 params.
                    elif param.type() == 'torch.cuda.FloatTensor':
                        fp32_params_this_group.append(param)
                        param_group['params'][i] = param

                    else:
                        raise TypeError(
                            'Wrapped parameters must be one of '
                            'torch.cuda.FloatTensor,  '
                            'torch.cuda.HalfTensor, or '
                            'torch.cuda.BFloat16Tensor. '
                            'Received {}'.format(param.type())
                        )

            self.float16_groups.append(float16_params_this_group)
            self.fp32_from_float16_groups.append(fp32_from_float16_params_this_group)
            self.fp32_from_fp32_groups.append(fp32_params_this_group)
        fp32_size = 0
        for groups in self.fp32_from_fp32_groups:
            fp32_size += len(groups)
        assert fp32_size == 0, "Not supported, because it is rarely used and makes code messy"

    def zero_grad(self, set_to_none=True):
        """We only need to zero the model related parameters, i.e.,
        float16_groups & fp32_from_fp32_groups. We additionally zero
        fp32_from_float16_groups as a memory optimization to reduce
        fragmentation; in the case of set_to_none==True, the space
        used by this field can be safely deallocated at this point."""
        for group in self.float16_groups:
            _zero_grad_group_helper(group, set_to_none)
        for group in self.fp32_from_float16_groups:
            _zero_grad_group_helper(group, set_to_none)
        for group in self.fp32_from_fp32_groups:
            _zero_grad_group_helper(group, set_to_none)

    def _release_grad_fp32_from_fp16(self, set_to_none=True):
        for group in self.fp32_from_float16_groups:
            _zero_grad_group_helper(group, set_to_none)

    def _collect_main_grad_data_for_unscaling(self):

        main_grads = []

        # fp32 params from float16 ones.
        for main_group in self.fp32_from_float16_groups:
            for main_param in main_group:
                if main_param.grad is not None:
                    main_grads.append(main_param.grad.data)

        # Append fp32 parameters.
        for main_group in self.fp32_from_fp32_groups:
            for main_param in main_group:
                if main_param.grad is not None:
                    main_grads.append(main_param.grad.data)

        return main_grads

    def _get_model_and_main_params_data_float16(self):
        model_data = []
        main_data = []
        for model_group, main_group in zip(self.float16_groups, self.fp32_from_float16_groups):
            for model_param, main_param in zip(model_group, main_group):
                model_data.append(model_param.data)
                main_data.append(main_param.data)
        return model_data, main_data

    def _copy_model_grads_to_main_grads(self):
        # This only needs to be done for the float16 group.
        for model_group, main_group in zip(self.float16_groups, self.fp32_from_float16_groups):
            for model_param, main_param in zip(model_group, main_group):
                if hasattr(model_param, 'main_grad'):
                    main_param.grad = model_param.main_grad.float()
                else:
                    if model_param.grad is not None:
                        main_param.grad = model_param.grad.float()

                # Safe to deallocate model's grad/main_grad after copying.
                # (If using contiguous buffers, main_grad's memory should
                # persist and therefore should not be deallocated.)
                model_param.grad = None

        # For fp32 grads, we need to reset the grads to main grad.
        for model_group in self.fp32_from_fp32_groups:
            for model_param in model_group:
                model_param.grad = model_param.main_grad

    def _copy_main_params_to_model_params(self):
        # Only needed for the float16 params.
        model_data, main_data = self._get_model_and_main_params_data_float16()
        _multi_tensor_copy_this_to_that(
            this=main_data, that=model_data, overflow_buf=self._dummy_overflow_buf
        )

    def _copy_model_params_to_main_params(self):
        # Only needed for the float16 params.
        model_data, main_data = self._get_model_and_main_params_data_float16()
        _multi_tensor_copy_this_to_that(
            this=model_data, that=main_data, overflow_buf=self._dummy_overflow_buf
        )

    def state_dict(self):
        state_dict = {}
        state_dict['optimizer'] = self.optimizer.state_dict()
        if self.grad_scaler:
            state_dict['grad_scaler'] = self.grad_scaler.state_dict()
        state_dict['fp32_from_fp16_params'] = self.fp32_from_float16_groups
        return state_dict

    def sharded_state_dict(
        self, model_sharded_state_dict: ShardedStateDict, is_loading: bool = False
    ):

        if is_loading:
            self.init_state_fn(self.optimizer)

        state_dict = self.state_dict()

        id_to_sharded_param_map = get_param_id_to_sharded_param_map(
            model_sharded_state_dict, chain.from_iterable(g for g in self.float16_groups)
        )

        # Convert fp32_from_fp16_params
        assert len(state_dict['fp32_from_fp16_params']) == len(
            state_dict['optimizer']['param_groups']
        )
        state_dict['fp32_from_fp16_params'] = [
            [
                make_sharded_optimizer_tensor(
                    id_to_sharded_param_map[param_id],
                    fp32_param,
                    prefix=f'optimizer.state.fp32_param',
                )
                for param_id, fp32_param in zip(state_group['params'], fp32_group)
            ]
            for fp32_group, state_group in zip(
                state_dict['fp32_from_fp16_params'], state_dict['optimizer']['param_groups']
            )
        ]

        step = self._extract_common_per_param_step(state_dict['optimizer'])

        # Convert regular optimizer state
        # all optimizer parameters passed to optim_state_to_sharding_state are
        # expected to have the same shape as the model parameters,
        # so we save the step separately and ignore it here
        optim_state_to_sharding_state(
            state_dict['optimizer'], id_to_sharded_param_map, exclude_keys="step"
        )
        # save step as a shared step among all parameters. Separate per-parameter
        # steps are not supported
        state_dict['optimizer']['state']['common_step'] = step
        return state_dict

    def load_state_dict(self, state_dict):
        pipeline_parallel_size = parallel_state.get_pipeline_model_parallel_world_size()
        # Optimizer.
        optimizer_key = 'optimizer'
        if optimizer_key not in state_dict:
            optimizer_key = 'optimizer_state_dict'
            logger.info('***WARNING*** loading optimizer from ' 'an old checkpoint ...')
        if 'common_step' in state_dict[optimizer_key]['state']:
            common_step = state_dict[optimizer_key]['state'].pop('common_step')
            self._restore_common_per_param_step(state_dict[optimizer_key], common_step)
        self.optimizer.load_state_dict(state_dict[optimizer_key])

        # Grad scaler.
        if 'grad_scaler' not in state_dict:
            if self.config.fp16:
                logger.info(
                    '***WARNING*** found an old checkpoint, will not ' 'load grad scaler ...'
                )
        else:
            if self.grad_scaler:
                self.grad_scaler.load_state_dict(state_dict['grad_scaler'])
            else:
                logger.info(
                    '***WARNING*** fould the grad scaler in the '
                    'checkpoint but it is None in the class. '
                    'Skipping loading grad scaler ...'
                )

        # Copy data for the main params.
        fp32_from_float16_params_key = 'fp32_from_fp16_params'
        if fp32_from_float16_params_key not in state_dict:
            fp32_from_float16_params_key = 'fp32_from_fp16'
        for current_group, saved_group in zip(
            self.fp32_from_float16_groups, state_dict[fp32_from_float16_params_key]
        ):
            for current_param, saved_param in zip(current_group, saved_group):
                current_param.data.copy_(saved_param.data)


class FP32Optimizer(MegatronOptimizer):
    """Float32 optimizer.

    Args:
        optimizer (torch.optim.Optimizer): base optimizer such as Adam or SGD.
        config (OptimizerConfig): configuration object for optimizer.
        init_state_fn (Callable, optional): function to initialize state in the optimizer.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        config: OptimizerConfig,
        init_state_fn: Callable,
    ):

        super(FP32Optimizer, self).__init__(
            optimizer,
            config,
            init_state_fn,
        )

        self._scale = torch.tensor([1.0], dtype=torch.float, device='cuda')

    def zero_grad(self, set_to_none=True):
        """Copied from torch.optim.optimizer"""
        for group in self.optimizer.param_groups:
            _zero_grad_group_helper(group['params'], set_to_none)

    def get_loss_scale(self):
        """FP32 optimizer does not do any scaling."""
        return self._scale

    @torch.no_grad()
    def prepare_grads(self) -> bool:
        """Pre-processing gradients before the optimizer step, returns whether inf/nan is found."""
        timers = self.config.timers

        # Copy main_grads to grads.
        if timers is not None:
            timers('optimizer-copy-to-main-grad', log_level=1).start(
                barrier=self.config.barrier_with_L1_time
            )
        for param_group in self.optimizer.param_groups:
            for param in param_group['params']:
                param.grad = param.main_grad
        if timers is not None:
            timers('optimizer-copy-to-main-grad').stop()

        return False

    @torch.no_grad()
    def step_with_ready_grads(self) -> bool:
        """Step the optimizer with ready gradients, return successful."""
        timers = self.config.timers

        # Update parameters.
        if timers is not None:
            timers('optimizer-inner-step', log_level=1).start(
                barrier=self.config.barrier_with_L1_time
            )
        self.optimizer.step()
        if timers is not None:
            timers('optimizer-inner-step').stop()

        return True

    @torch.no_grad()
    def step(self):
        """Clip gradients (if needed) and step the base optimizer.
        Always return successful since there is no overflow."""
        timers = self.config.timers

        found_inf_flag = self.prepare_grads()
        if found_inf_flag:
            return False, None, None

        # Clip gradients.
        if timers is not None:
            timers('optimizer-clip-main-grad', log_level=1).start(
                barrier=self.config.barrier_with_L1_time
            )
        grad_norm = None
        if self.config.clip_grad > 0.0:
            grad_norm = self.clip_grad_norm(self.config.clip_grad)
        if timers is not None:
            timers('optimizer-clip-main-grad').stop()

        # Count the zeros in the grads.
        if timers is not None:
            timers('optimizer-count-zeros', log_level=1).start(
                barrier=self.config.barrier_with_L1_time
            )
        num_zeros_in_grad = self.count_zeros() if self.config.log_num_zeros_in_grad else None
        if timers is not None:
            timers('optimizer-count-zeros').stop()

        success = self.step_with_ready_grads()

        # No overflow for FP32 optimizer.
        return success, grad_norm, num_zeros_in_grad

    def reload_model_params(self):
        pass

    def state_dict(self):
        return self.optimizer.state_dict()

    def load_state_dict(self, state_dict):
        pipeline_parallel_size = parallel_state.get_pipeline_model_parallel_world_size()
        if 'common_step' in state_dict['state']:
            common_step = state_dict['state'].pop('common_step')
            self._restore_common_per_param_step(state_dict, common_step)
        self.optimizer.load_state_dict(state_dict)

    def sharded_state_dict(
        self, model_sharded_state_dict: ShardedStateDict, is_loading: bool = False
    ):
        if is_loading:
            self.init_state_fn(self.optimizer)

        state_dict = self.state_dict()
        id_to_sharded_param_map = get_param_id_to_sharded_param_map(
            model_sharded_state_dict, self.get_parameters()
        )
        step = self._extract_common_per_param_step(state_dict)

        # all optimizer parameters passed to optim_state_to_sharding_state are
        # expected to have the same shape as the model parameters,
        # so we save the step separately and ignore it here
        optim_state_to_sharding_state(state_dict, id_to_sharded_param_map, exclude_keys="step")
        # save step as a shared step among all parameters. Separate per-parameter
        # steps are not supported
        state_dict['state']['common_step'] = step
        return state_dict


class ProxyDict:
    """
    A dictionary-like object that proxies to a list of dictionaries.

    e.g., ProxyDict([{'a': 1}, {'b': 2}]) behaves like:
    {
        (0, 'a'): 1,
        (1, 'b'): 2,
    }
    We use tuples as keys to avoid ambiguity with the keys of the inner dicts.
    """

    def __init__(self, inner_dicts: List[dict]):
        self._inner_dicts = inner_dicts

    def __getitem__(self, key: Tuple[int, str]):
        idx, inner_key = key
        return self._inner_dicts[idx].get(inner_key)

    def __setitem__(self, key: Tuple[int, str], value: Any):
        idx, inner_key = key
        self._inner_dicts[idx][inner_key] = value

    def __len__(self) -> int:
        return sum([len(inner_dict) for inner_dict in self._inner_dicts])

    def __iter__(self):
        for idx, inner_dict in enumerate(self._inner_dicts):
            for inner_key in inner_dict:
                yield (idx, inner_key)

    def items(self):
        for idx, inner_dict in enumerate(self._inner_dicts):
            for inner_key, value in inner_dict.items():
                yield (idx, inner_key), value


class ChainedOptimizer(MegatronOptimizer):
    """ChainedOptimizer is designed for a collection of optimizers.

    These optimizers are responsible for different parts of multiple models for
    a training task and will be executed one-by-one when the model is updated.

    Args:
        chained_optimizers: a list of optimizers.
    """

    def __init__(self, chained_optimizers: List[MegatronOptimizer]):
        self.chained_optimizers = chained_optimizers

    @property
    def param_groups(self) -> List[dict]:
        param_groups = []
        for optimizer in self.chained_optimizers:
            param_groups += optimizer.param_groups
        return param_groups

    @property
    def state(self) -> ProxyDict:
        """
        Return optimizer state with tuple keys, where the first element is the
        index of the optimizer in the list of chained optimizers.
        """
        return ProxyDict([opt.state for opt in self.chained_optimizers])

    def zero_grad(self, set_to_none=True):
        for optimizer in self.chained_optimizers:
            optimizer.zero_grad(set_to_none)

    def get_loss_scale(self):
        return self.chained_optimizers[0].get_loss_scale()

    def reload_model_params(self):
        for optimizer in self.chained_optimizers:
            optimizer.reload_model_params()

    def state_dict(self):
        return [optimizer.state_dict() for optimizer in self.chained_optimizers]

    def sharded_state_dict(
        self, model_sharded_state_dict: ShardedStateDict, is_loading: bool = False, **kwargs
    ):
        sharded_state_dict = {}
        for optimizer_idx, optimizer in enumerate(self.chained_optimizers):
            optim_state_dict = optimizer.sharded_state_dict(
                model_sharded_state_dict, is_loading, **kwargs
            )
            add_prefix_for_sharding(optim_state_dict, f'chained_{optimizer_idx}.')
            sharded_state_dict[optimizer_idx] = optim_state_dict
        return sharded_state_dict

    def load_state_dict(self, state_dict):
        if len(self.chained_optimizers) != len(state_dict):
            raise RuntimeError(
                f'Expected {len(self.chained_optimizers)} entries'
                f' in state dict, but got {len(state_dict)}.'
            )
        if isinstance(state_dict, dict):
            state_dict = (v for k, v in sorted(state_dict.items()))
        for optimizer, state in zip(self.chained_optimizers, state_dict):
            optimizer.load_state_dict(state)

    @torch.no_grad()
    def prepare_grads(self) -> bool:
        """Pre-processing gradients before the optimizer step, returns whether inf/nan is found."""
        found_inf_flag = False
        for optimizer in self.chained_optimizers:
            found_inf_flag |= optimizer.prepare_grads()

        return found_inf_flag

    @torch.no_grad()
    def step_with_ready_grads(self) -> bool:
        """Step the optimizer with ready gradients, return successful."""
        success = True
        for optimizer in self.chained_optimizers:
            success &= optimizer.step_with_ready_grads()

        return success

    def disable_pre_hook(self):
        for optimizer in self.chained_optimizers:
            if (
                not optimizer.config.use_distributed_optimizer
                or not optimizer.config.overlap_param_gather
            ):
                raise ValueError(
                    "disable_pre_hook should only be called with 'use_distributed_optimizer' "
                    "and 'overlap_param_gather' both enabled."
                )
            optimizer.disable_pre_hook()

    def enable_pre_hook(self):
        for optimizer in self.chained_optimizers:
            if (
                not optimizer.config.use_distributed_optimizer
                or not optimizer.config.overlap_param_gather
            ):
                raise ValueError(
                    "enable_pre_hook should only be called with 'use_distributed_optimizer' "
                    "and 'overlap_param_gather' both enabled."
                )
            optimizer.enable_pre_hook()

    @torch.no_grad()
    def step(self):
        """ChainedOptimizer will step all optimizers one by one."""
        found_inf_flag = self.prepare_grads()
        if found_inf_flag:
            return False, None, None

        # Get grad norm.
        grad_norms = []
        for optimizer in self.chained_optimizers:
            _grad_norm = optimizer.get_grad_norm()
            grad_norms += [_grad_norm if _grad_norm else 0.0]
        grad_norm = math.sqrt(sum([x**2 for x in grad_norms]))

        # Clip gradients.
        for optimizer in self.chained_optimizers:
            if optimizer.config.clip_grad > 0.0:
                clip_grad_by_total_norm_fp32(
                    optimizer.get_parameters(),
                    max_norm=optimizer.config.clip_grad,
                    total_norm=grad_norm,
                )

        # Count the zeros in the grads.
        num_zeros_in_grad = 0
        for optimizer in self.chained_optimizers:
            num_zeros_in_grad += (
                optimizer.count_zeros() if optimizer.config.log_num_zeros_in_grad else 0
            )

        update_successful = self.step_with_ready_grads()

        return update_successful, grad_norm, num_zeros_in_grad

    def save_parameter_state(self, filename: str):
        """Save the distributed parameter states of all optimizers to a file.

        Args:
            filename (str): path to save parameter state to.
        """
        save_states = False
        states = []
        for optimizer in self.chained_optimizers:
            if hasattr(optimizer, 'get_parameter_state_dp_zero'):
                state_dict = optimizer.get_parameter_state_dp_zero()

                # Save checkpoint economically, only when DP rank = 0, state dict
                # needs to be saved.
                if torch.distributed.get_rank(optimizer.data_parallel_group) == 0:
                    states.append(state_dict)
                    save_states = True
                else:
                    states.append(None)
            else:
                states.append(None)

        if save_states:
            torch.save(states, filename)

    def load_parameter_state(self, filename: str):
        """Load the distributed parameter states of all optimizers from a file.

        Args:
            filename (str): path to load parameter state from.
        """
        states = None
        for idx, optimizer in enumerate(self.chained_optimizers):
            if not hasattr(optimizer, 'load_parameter_state_from_dp_zero'):
                continue

            # Lazy loading checkpoint, state dict is needed only when DP rank = 0.
            if torch.distributed.get_rank(optimizer.data_parallel_group) == 0 and states is None:
                states = torch.load(filename)

            state_dict = states[idx] if states else None
            optimizer.load_parameter_state_from_dp_zero(state_dict)

    def finish_param_sync(self, model_index: int):
        """Finish parameter synchronization for all optimizers."""
        for optimizer in self.chained_optimizers:
            optimizer.finish_param_sync(model_index)
